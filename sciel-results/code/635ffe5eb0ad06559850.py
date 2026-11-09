import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    total_cells = n * m * p  # 120
    
    # Schoolbook fallback
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
    
    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result
    
    # Corrected Strassen coefficients
    strassen_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 0, 0, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 1, 0, 0],  # M6
        [1, 0, 0, 1]   # M7
    ]
    
    # Partition:
    # Part 1: 4x4x4 (i=0..3, j=0..3, k=0..3) -> 8 Strassen blocks (rank 56)
    # Part 2: 4x4x2 (i=0..3, j=0..3, k=4..5) -> 4 Strassen blocks (rank 28)
    # Part 3: 4x1x6 (i=0..3, j=4, k=0..5)   -> Schoolbook (rank 24)
    # Total rank: 56 + 28 + 24 = 108
    
    total_rank = 108
    
    if total_rank >= total_cells:
        return best_result
    
    U_new = np.zeros((n * m, total_rank), dtype=object)
    V_new = np.zeros((m * p, total_rank), dtype=object)
    W_new = np.zeros((p * n, total_rank), dtype=object)
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(U, V, W, col, i_base, j_base, k_base):
        for t in range(7):
            # u vector
            for ai in range(2):
                for aj in range(2):
                    local_idx = ai * 2 + aj
                    coeff = strassen_u[t][local_idx]
                    if coeff != 0:
                        i_glob = i_base + ai
                        j_glob = j_base + aj
                        U[i_glob * m + j_glob, col] = coeff
            
            # v vector
            for bj_local in range(2):
                for bk_local in range(2):
                    local_idx = bj_local * 2 + bk_local
                    coeff = strassen_v[t][local_idx]
                    if coeff != 0:
                        j_glob = j_base + bj_local
                        k_glob = k_base + bk_local
                        V[j_glob * p + k_glob, col] = coeff
            
            # w vector
            for ci in range(2):
                for ck in range(2):
                    local_idx = ci * 2 + ck
                    coeff = strassen_w[t][local_idx]
                    if coeff != 0:
                        i_glob = i_base + ci
                        k_glob = k_base + ck
                        W[k_glob * n + i_glob, col] = coeff
            
            col += 1
        return col
    
    # Part 1: 4x4x4
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                col = add_strassen_block(U_new, V_new, W_new, col, 2*bi, 2*bj, 2*bk)
    
    # Part 2: 4x4x2 (k=4,5)
    for bi in range(2):
        for bj in range(2):
            col = add_strassen_block(U_new, V_new, W_new, col, 2*bi, 2*bj, 4)
    
    # Part 3: 4x1x6 (j=4)
    for i in range(n):
        for k in range(p):
            U_new[i * m + 4, col] = 1
            V_new[4 * p + k, col] = 1
            W_new[k * n + i, col] = 1
            col += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
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
