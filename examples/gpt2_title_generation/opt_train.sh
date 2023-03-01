mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

export ET=deepspeed+mpu
SCRIPT_FILE=train_multi_gpu.py

echo "MODE_LNAME: $MN"
echo "ENV_TYPE: $ET"
echo "BATCH_SIZE: $BS"
echo "Model_Parallel_Size: $MP"
echo "Node: $NODE"
echo "GPU_NUM: $GPU_NUM"
echo "SCRIPT_FILE: $SCRIPT_FILE"

echo "PYTHONPATH: $PYTHONPATH"
echo "EPOCH_NUM: $EPOCH_NUM"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "pwd:" $(pwd)

unset KUBERNETES_PORT;


python $SCRIPT_FILE 2>&1 | tee log/opt-$MN.log.$now

#  MM="opt-125m-en" BATCH_SIZE=2 MP=2  NODE=1 GPU_NUM=2 sh opt_train.sh 


# python -m torch.distributed.launch \
#     --nproc_per_node $GPU_NUM --nnodes $WORLD_SIZE --node_rank $RANK \
#     --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#     $SCRIPT_FILE --not_call_launch \
#     --env_type $ET \
#     --batch_size $BATCH_SIZE \
#     --num_gpus $GPU_NUM \
#     --num_nodes $NODE \
#     --model_parallel_size $MP \
#     2>&1 | tee log/xxx.log.$now