"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Precompute index arrays for NumPy
    # For each (i,j,k), we need indices into U, V, W
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    # Try to find better decompositions
    for target_rank in range(25, rank_school):
        if time.time() - start_time > deadline - 20:
            break
        
        found = False
        for restart in range(15):
            if time.time() - start_time > deadline - 20:
                break
            
            # Initialize with structured pattern
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Each column r handles a specific (i,k) pair
            for r in range(target_rank):
                idx = r % (n * p)
                i0 = idx // p
                k0 = idx % p
                
                W[k0 * n + i0, r] = 1.0
                for j in range(m):
                    U[i0 * m + j, r] = 1.0
                    V[j * p + k0, r] = 1.0
            
            # Local search using NumPy
            def compute_err(U, V, W):
                # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                vals = np.zeros(len(i_idx))
                for t, (i, j, k) in enumerate(zip(i_idx, j_idx, k_idx)):
                    vals[t] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                return np.sum((vals - 1.0) ** 2)
            
            current_err = compute_err(U, V, W)
            
            for step in range(200):
                if time.time() - start_time > deadline - 15:
                    break
                if current_err < 1e-12:
                    break
                
                # Random modification
                mat = random.choice(['U', 'V', 'W'])
                col_idx = random.randint(0, target_rank - 1)
                
                if mat == 'U':
                    row = random.randint(0, n * m - 1)
                    old = U[row, col_idx]
                    U[row, col_idx] = old + random.choice([-2.0, -1.0, 1.0, 2.0])
                elif mat == 'V':
                    row = random.randint(0, m * p - 1)
                    old = V[row, col_idx]
                    V[row, col_idx] = old + random.choice([-2.0, -1.0, 1.0, 2.0])
                else:
                    row = random.randint(0, p * n - 1)
                    old = W[row, col_idx]
                    W[row, col_idx] = old + random.choice([-2.0, -1.0, 1.0, 2.0])
                
                new_err = compute_err(U, V, W)
                
                if new_err > current_err * 1.02:
                    # Revert
                    if mat == 'U':
                        U[row, col_idx] = old
                    elif mat == 'V':
                        V[row, col_idx] = old
                    else:
                        W[row, col_idx] = old
                else:
                    current_err = new_err
            
            if current_err < 1e-12:
                # Convert to lists and verify
                U_list = [[U[i, r] for r in range(target_rank)] for i in range(n * m)]
                V_list = [[V[i, r] for r in range(target_rank)] for i in range(m * p)]
                W_list = [[W[i, r] for r in range(target_rank)] for i in range(p * n)]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                val += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                            if abs(val - 1) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
