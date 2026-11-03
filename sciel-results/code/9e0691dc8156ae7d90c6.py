import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    Uses Strassen's algorithm recursively to achieve rank 56.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_strassen = np.zeros((4, 7), dtype=int)
    V_strassen = np.zeros((4, 7), dtype=int)
    W_strassen = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1

    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1

    # M3 = A00(B01-B11) -> C01 += M3
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[1, 2] = 1

    # M4 = A11(B10-B00) -> C10 += M4, C11 += M4
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[2, 3] = 1; W_strassen[3, 3] = 1

    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[1, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1

    rank_block = 7
    num_blocks = (m // 2) * (p // 2) # 2 * 4 = 8
    total_rank = rank_block * num_blocks # 56

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    for b1 in range(m // 2):
        for b2 in range(p // 2):
            block_idx = b1 * (p // 2) + b2
            col_offset = block_idx * rank_block
            
            for s in range(rank_block):
                # Local indices in 2x2x2 block
                # U: i in 0,1; j_local in 0,1
                for i in range(2):
                    for j_local in range(2):
                        val_u = U_strassen[i * 2 + j_local, s]
                        if val_u != 0:
                            global_i = i
                            global_j = b1 * 2 + j_local
                            U[global_i * m + global_j, col_offset + s] += val_u
                
                # V: j_local in 0,1; k_local in 0,1
                for j_local in range(2):
                    for k_local in range(2):
                        val_v = V_strassen[j_local * 2 + k_local, s]
                        if val_v != 0:
                            global_j = b1 * 2 + j_local
                            global_k = b2 * 2 + k_local
                            V[global_j * p + global_k, col_offset + s] += val_v
                
                # W: k_local in 0,1; i in 0,1
                # W index in strassen: k_local * 2 + i
                for k_local in range(2):
                    for i in range(2):
                        val_w = W_strassen[k_local * 2 + i, s]
                        if val_w != 0:
                            global_k = b2 * 2 + k_local
                            global_i = i
                            W[global_k * n + global_i, col_offset + s] += val_w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
