import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    
    Uses a block-partitioning approach:
    - Row 0 (1x4x8): Naive multiplication with rank 32
    - Rows 1-2, Cols 0-3 (2x4x4): Strassen-based decomposition with rank 28
    - Rows 1-2, Cols 4-7 (2x4x4): Strassen-based decomposition with rank 28
    
    Total rank: 32 + 28 + 28 = 88
    """
    n, m, p = 3, 4, 8
    total_rank = 88
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Corrected Strassen vectors for 2x2x2 matrix multiplication
    # Convention: U index = i*2+j, V index = j*2+k, W index = k*2+i
    # 
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U vectors (A factors): [A11, A12, A21, A22]
    U_vecs = [
        [1,  0,  0,  1],   # M1: A11 + A22
        [0,  0,  1,  1],   # M2: A21 + A22
        [1,  0,  0,  0],   # M3: A11
        [0,  0,  0,  1],   # M4: A22
        [1,  1,  0,  0],   # M5: A11 + A12
        [-1, 0,  1,  0],   # M6: A21 - A11
        [0,  1,  0, -1]    # M7: A12 - A22
    ]
    # V vectors (B factors): [B11, B12, B21, B22]
    V_vecs = [
        [1,  0,  0,  1],   # M1: B11 + B22
        [1,  0,  0,  0],   # M2: B11
        [0,  0,  1, -1],   # M3: B21 - B22
        [1, -1, 0,  0],    # M4: B11 - B12
        [0,  0,  0,  1],   # M5: B22
        [1,  1,  0,  0],   # M6: B11 + B12
        [0,  0,  1,  1]    # M7: B21 + B22
    ]
    # W vectors (C factors): [C11, C12, C21, C22] with index k*2+i
    W_vecs = [
        [1,  0,  0,  1],   # M1: C11 + C22
        [0,  0,  1,  0],   # M2: -C21
        [0,  1,  0,  0],   # M3: C12 + C22
        [1,  0,  1,  0],   # M4: C11 + C21
        [-1, 1,  0,  0],   # M5: -C11 + C12
        [0,  0,  0,  1],   # M6: C22
        [1,  0,  0, -1]    # M7: C11 - C22
    ]
    
    def add_strassen_block(i_off, j_off, k_off, col_start):
        """
        Add a 2x2x2 Strassen block to the decomposition.
        
        i_off: global row offset for A (0 or 1)
        j_off: global col offset for A / row offset for B (0, 2, or 4)
        k_off: global col offset for B / row offset for C (0, 2, 4, or 6)
        col_start: starting column in U, V, W matrices
        """
        for t in range(7):
            r = col_start + t
            # Fill U: index = (i_off + local_i) * m + (j_off + local_j)
            for local_i in range(2):
                for local_j in range(2):
                    val = U_vecs[t][local_i * 2 + local_j]
                    if val != 0:
                        global_i = i_off + local_i
                        global_j = j_off + local_j
                        U[global_i * m + global_j, r] = val
            # Fill V: index = (j_off + local_j) * p + (k_off + local_k)
            for local_j in range(2):
                for local_k in range(2):
                    val = V_vecs[t][local_j * 2 + local_k]
                    if val != 0:
                        global_j = j_off + local_j
                        global_k = k_off + local_k
                        V[global_j * p + global_k, r] = val
            # Fill W: index = (k_off + local_k) * n + (i_off + local_i)
            for local_k in range(2):
                for local_i in range(2):
                    val = W_vecs[t][local_k * 2 + local_i]
                    if val != 0:
                        global_k = k_off + local_k
                        global_i = i_off + local_i
                        W[global_k * n + global_i, r] = val
    
    # Part 1: Row 0 (1x4x8) - Naive multiplication
    # C[0,k] = sum_j A[0,j] * B[j,k]
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    
    # Part 2: Rows 1-2, Cols 0-3 (2x4x4)
    # Split A into left (cols 0-1) and right (cols 2-3)
    # Split B into left (cols 0-1) and right (cols 2-3)
    # C_L = A_L @ B_LL + A_R @ B_RL
    # C_R = A_L @ B_LR + A_R @ B_RR
    
    # C_L: A_L (i_off=1, j_off=0) * B_LL (j_off=0, k_off=0)
    add_strassen_block(1, 0, 0, col)
    col += 7
    
    # C_L: A_R (i_off=1, j_off=2) * B_RL (j_off=2, k_off=0)
    add_strassen_block(1, 2, 0, col)
    col += 7
    
    # C_R: A_L (i_off=1, j_off=0) * B_LR (j_off=0, k_off=2)
    add_strassen_block(1, 0, 2, col)
    col += 7
    
    # C_R: A_R (i_off=1, j_off=2) * B_RR (j_off=2, k_off=2)
    add_strassen_block(1, 2, 2, col)
    col += 7
    
    # Part 3: Rows 1-2, Cols 4-7 (2x4x4)
    # B sub-blocks for right half:
    # B_LL_right: j_off=0, k_off=4
    # B_LR_right: j_off=0, k_off=6
    # B_RL_right: j_off=2, k_off=4
    # B_RR_right: j_off=2, k_off=6
    
    # C_L_right: A_L (i_off=1, j_off=0) * B_LL_right (j_off=0, k_off=4)
    add_strassen_block(1, 0, 4, col)
    col += 7
    
    # C_L_right: A_R (i_off=1, j_off=2) * B_RL_right (j_off=2, k_off=4)
    add_strassen_block(1, 2, 4, col)
    col += 7
    
    # C_R_right: A_L (i_off=1, j_off=0) * B_LR_right (j_off=0, k_off=6)
    add_strassen_block(1, 0, 6, col)
    col += 7
    
    # C_R_right: A_R (i_off=1, j_off=2) * B_RR_right (j_off=2, k_off=6)
    add_strassen_block(1, 2, 6, col)
    col += 7
    
    return {"U": U, "V": V, "W": W}
