import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave safety margin before 120s limit
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Verify function
    def verify_decomposition(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Strategy: Decompose into two M_{2,2,7} blocks
    # A = [A1, A2], B = [B1; B2]
    # C = A1 B1 + A2 B2
    # Find rank r decomposition for M_{2,2,7}
    # Then full rank is 2*r
    
    n_sub = 2
    m_sub = 2
    p_sub = 7
    
    # Search for M_{2,2,7}
    found_sub = False
    best_sub_rank = 28 # schoolbook for sub
    best_sub_U = None
    best_sub_V = None
    best_sub_W = None
    
    # Schoolbook for sub
    U_sub_school = [[0] * 28 for _ in range(4)]
    V_sub_school = [[0] * 28 for _ in range(14)]
    W_sub_school = [[0] * 28 for _ in range(14)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U_sub_school[i * 2 + j][col] = 1
                V_sub_school[j * 7 + k][col] = 1
                W_sub_school[k * 2 + i][col] = 1
                col += 1
    best_sub_U = U_sub_school
    best_sub_V = V_sub_school
    best_sub_W = W_sub_school
    best_sub_rank = 28
    
    # Try to find lower rank for sub-problem
    for r in range(14, 28):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(r * 100 + attempt)
            
            # Random sparse U, V with entries -1, 0, 1
            U_sub = np.zeros((4, r), dtype=int)
            V_sub = np.zeros((14, r), dtype=int)
            
            for c in range(r):
                # Set some non-zeros
                nz_U = np.random.choice(4, size=np.random.randint(1, 3), replace=False)
                for idx in nz_U:
                    U_sub[idx, c] = np.random.choice([-1, 1])
                    
                nz_V = np.random.choice(14, size=np.random.randint(1, 4), replace=False)
                for idx in nz_V:
                    V_sub[idx, c] = np.random.choice([-1, 1])
            
            # Build linear system for W
            # Equations: sum_c U[i, c] V[j, c] W[k, c] = 1 for all i, j, k
            # i in 0..1, j in 0..1, k in 0..6
            # W has 14 rows (k*2+i) and r columns
            # Unknowns: W_flat of size 14*r
            
            A = np.zeros((4 * 7, 14 * r), dtype=int) # 28 equations
            b = np.ones(28, dtype=int)
            
            eq_idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(7):
                        for c in range(r):
                            coeff = U_sub[i * 2 + j, c] * V_sub[j * 7 + k, c]
                            if coeff != 0:
                                # W index: k*2+i, col c
                                # Flat index: (k*2+i)*r + c
                                A[eq_idx, (k * 2 + i) * r + c] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sub = W_flat.reshape(14, r)
                
                # Check if W_sub is integer or half-integer
                W_int = np.round(W_sub * 2)
                if np.allclose(W_int / 2, W_sub, atol=1e-10):
                    # Verify
                    U_list = U_sub.tolist()
                    V_list = V_sub.tolist()
                    W_list = [[W_int[i, c] / 2 for c in range(r)] for i in range(14)]
                    
                    if verify_decomposition(U_list, V_list, W_list, r, 2, 2, 7):
                        best_sub_rank = r
                        best_sub_U = U_list
                        best_sub_V = V_list
                        best_sub_W = W_list
                        found_sub = True
                        break
            except Exception:
                pass
        
        if found_sub:
            break
    
    # If we found a sub-solution, construct the full solution
    if found_sub and best_sub_rank < 28:
        r = best_sub_rank
        full_rank = 2 * r
        
        U_full = [[0] * full_rank for _ in range(n * m)] # 8 x 2r
        V_full = [[0] * full_rank for _ in range(m * p)] # 28 x 2r
        W_full = [[0] * full_rank for _ in range(p * n)] # 14 x 2r
        
        # Block 1: j in {0, 1}
        for i in range(2):
            for j in range(2):
                for c in range(r):
                    U_full[i * 4 + j][c] = best_sub_U[i * 2 + j][c]
        
        for j in range(2):
            for k in range(7):
                for c in range(r):
                    V_full[j * 7 + k][c] = best_sub_V[j * 7 + k][c]
        
        # Block 2: j in {2, 3}
        for i in range(2):
            for j in range(2, 4):
                for c in range(r):
                    U_full[i * 4 + j][r + c] = best_sub_U[i * 2 + (j - 2)][c]
        
        for j in range(2, 4):
            for k in range(7):
                for c in range(r):
                    V_full[j * 7 + k][r + c] = best_sub_V[(j - 2) * 7 + k][c]
        
        # W is the same for both blocks
        for k in range(7):
            for i in range(2):
                for c in range(r):
                    W_full[k * 2 + i][c] = best_sub_W[k * 2 + i][c]
                    W_full[k * 2 + i][r + c] = best_sub_W[k * 2 + i][c]
        
        # Verify full solution
        if verify_decomposition(U_full, V_full, W_full, full_rank, n, m, p):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = full_rank
    
    return best_result
