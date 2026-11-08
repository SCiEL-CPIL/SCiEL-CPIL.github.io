import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 definitions
    # U, V, W are lists of (index, value)
    strassen_U = [
        [(0, 1), (3, 1)], # r0
        [(2, 1), (3, 1)], # r1
        [(0, 1)],         # r2
        [(3, 1)],         # r3
        [(0, 1), (1, 1)], # r4
        [(2, 1), (0, -1)],# r5
        [(1, 1), (3, -1)] # r6
    ]
    strassen_V = [
        [(0, 1), (3, 1)], # r0
        [(0, 1)],         # r1
        [(1, 1), (3, -1)],# r2
        [(2, 1), (0, -1)],# r3
        [(3, 1)],         # r4
        [(0, 1), (1, 1)], # r5
        [(2, 1), (3, 1)]  # r6
    ]
    strassen_W = [
        [(0, 1), (3, 1)], # r0
        [(1, 1), (3, -1)],# r1
        [(2, 1), (3, 1)], # r2
        [(0, 1), (1, 1)], # r3
        [(0, -1), (2, 1)],# r4
        [(3, 1)],         # r5
        [(0, 1)]          # r6
    ]
    
    def add_strassen_block(base_rank, i_offset, j_offset):
        for r in range(7):
            current_rank = base_rank + r
            
            # U: i in 0..1, j in 0..1 (local)
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            # V: j in 0..1 (local), k in 0..7
            # V is flattened j*8 + k
            for idx, val in strassen_V[r]:
                local_j, local_k_placeholder = divmod(idx, 2) # idx is 0..3, so local_j is 0..1
                global_j = local_j + j_offset
                # The value applies to all k
                for k in range(8):
                    V[global_j * 8 + k][current_rank] += val
            
            # W: k in 0..7, i in 0..1 (local)
            # W is flattened k*2 + i
            for idx, val in strassen_W[r]:
                local_k_placeholder, local_i = divmod(idx, 2) # idx is 0..3, so local_i is 0..1
                global_i = local_i + i_offset
                # The value applies to all k
                for k in range(8):
                    W[k * 2 + global_i][current_rank] += val

    # Block 1: j_offset = 0
    add_strassen_block(0, 0, 0)
    
    # Block 2: j_offset = 2
    add_strassen_block(7, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
