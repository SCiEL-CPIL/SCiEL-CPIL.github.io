"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    total_cells = n * m * p
    
    # Initialize with schoolbook rank 120 as a fallback
    U_school = np.zeros((n * m, total_cells), dtype=object)
    V_school = np.zeros((m * p, total_cells), dtype=object)
    W_school = np.zeros((p * n, total_cells), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "rank": total_cells
    }
    
    # Check deadline
    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result
    
    # Strassen-based decomposition for 2x2x2 blocks
    # For each 2x2x2 block, we use 7 rank-1 tensors
    # The 7 terms are defined by their u, v, w vectors
    
    # Define the 7 Strassen terms as (u_vec, v_vec, w_vec) for 2x2x2
    # u_vec: 4 entries for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_vec: 4 entries for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_vec: 4 entries for C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_u = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: -A00 + A10
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10 - B11
        [-1, 1, 0, 0], # M4: -B00 + B01
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    strassen_w = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, 0],  # M2: C10
        [0, 1, 0, -1], # M3: C01 - C11
        [-1, 0, 1, 0], # M4: -C00 + C10
        [0, 1, 0, 1],  # M5: C01 + C11
        [1, 0, 0, 0],  # M6: C00
        [-1, 0, 0, 1]  # M7: -C00 + C11
    ]
    
    # Partition: n=4 -> 2 blocks of 2, m=5 -> 2 blocks of 2 + 1, p=6 -> 3 blocks of 2
    num_i_blocks = n // 2  # 2
    num_j_blocks = m // 2  # 2
    num_k_blocks = p // 2  # 3
    remaining_j = m - num_j_blocks * 2  # 1
    
    total_rank = num_i_blocks * num_j_blocks * num_k_blocks * 7 + remaining_j * n * p
    # = 2*2*3*7 + 1*4*6 = 84 + 24 = 108
    
    if total_rank >= total_cells:
        return best_result  # No improvement
    
    U_new = np.zeros((n * m, total_rank), dtype=object)
    V_new = np.zeros((m * p, total_rank), dtype=object)
    W_new = np.zeros((p * n, total_rank), dtype=object)
    
    col = 0
    
    # Process Strassen blocks
    for bi in range(num_i_blocks):
        for bj in range(num_j_blocks):
            for bk in range(num_k_blocks):
                for t in range(7):
                    # u vector: 4 entries for local A indices (0,0), (0,1), (1,0), (1,1)
                    for ai in range(2):
                        for aj in range(2):
                            local_idx = ai * 2 + aj
                            coeff = strassen_u[t][local_idx]
                            if coeff != 0:
                                i_glob = 2 * bi + ai
                                j_glob = 2 * bj + aj
                                U_new[i_glob * m + j_glob, col] = coeff
                    
                    # v vector: 4 entries for local B indices (0,0), (0,1), (1,0), (1,1)
                    for bj_local in range(2):
                        for bk_local in range(2):
                            local_idx = bj_local * 2 + bk_local
                            coeff = strassen_v[t][local_idx]
                            if coeff != 0:
                                j_glob = 2 * bj + bj_local
                                k_glob = 2 * bk + bk_local
                                V_new[j_glob * p + k_glob, col] = coeff
                    
                    # w vector: 4 entries for local C indices (0,0), (0,1), (1,0), (1,1)
                    for ci in range(2):
                        for ck in range(2):
                            local_idx = ci * 2 + ck
                            coeff = strassen_w[t][local_idx]
                            if coeff != 0:
                                i_glob = 2 * bi + ci
                                k_glob = 2 * bk + ck
                                W_new[k_glob * n + i_glob, col] = coeff
                    
                    col += 1
    
    # Process remaining j indices (j = num_j_blocks*2 to m-1)
    for j in range(num_j_blocks * 2, m):
        for i in range(n):
            for k in range(p):
                U_new[i * m + j, col] = 1
                V_new[j * p + k, col] = 1
                W_new[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # This should equal 1 for all i,j,k (schoolbook)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for t in range(rank):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if total != 1:
                        return False
        return True
    
    if verify(U_new, V_new, W_new, n, m, p):
        best_result = {
            "U": U_new.tolist(),
            "V": V_new.tolist(),
            "W": W_new.tolist(),
            "rank": total_rank
        }
    
    return best_result
# EVOLVE-BLOCK-END
