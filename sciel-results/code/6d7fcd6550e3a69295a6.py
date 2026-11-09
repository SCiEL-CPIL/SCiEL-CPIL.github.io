"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We construct the schoolbook decomposition directly in list format as required
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n*m, m*p, p*n)
    
    # Try to find lower rank decompositions
    # We iterate R from min_rank up to schoolbook
    for R in range(min_rank, rank_school + 1):
        if time.time() > deadline:
            break
            
        # If we already have a solution with rank <= R, we could stop, 
        # but we are iterating upwards, so the first valid R is the best.
        # However, if we start from min_rank, the first success is the minimum.
        
        # Multiple random restarts
        # Reduce restarts as R increases to save time if lower ranks fail?
        # Or keep it constant? Let's use 100 restarts.
        max_restarts = 100
        max_iters = 500
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization in 3D shapes for easier einsum
            # U: (n, m, R), V: (m, p, R), W: (p, n, R)
            U = np.random.uniform(-1, 1, (n, m, R))
            V = np.random.uniform(-1, 1, (m, p, R))
            W = np.random.uniform(-1, 1, (p, n, R))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                # M[i,j,r] = sum_k V[j,k,r] * W[k,i,r]
                # V is (m, p, R), W is (p, n, R)
                # We want result (n, m, R)
                # M[i, j, r] = sum_k V[j, k, r] W[k, i, r]
                # Using einsum: 'jkr,kir->ijr'
                M = np.einsum('jkr,kir->ijr', V, W)
                norm_sq = np.sum(M**2, axis=2, keepdims=True)
                # Avoid division by zero
                norm_sq[norm_sq < 1e-12] = 1.0
                U = M / norm_sq
                
                # Update V
                # N[j,k,r] = sum_i U[i,j,r] * W[k,i,r]
                # U is (n, m, R), W is (p, n, R)
                # We want result (m, p, R)
                # N[j, k, r] = sum_i U[i, j, r] W[k, i, r]
                # Using einsum: 'ijr,kir->jkr'
                N = np.einsum('ijr,kir->jkr', U, W)
                norm_sq = np.sum(N**2, axis=2, keepdims=True)
                norm_sq[norm_sq < 1e-12] = 1.0
                V = N / norm_sq
                
                # Update W
                # O[k,i,r] = sum_j U[i,j,r] * V[j,k,r]
                # U is (n, m, R), V is (m, p, R)
                # We want result (p, n, R)
                # O[k, i, r] = sum_j U[i, j, r] V[j, k, r]
                # Using einsum: 'ijr,jkr->kir'
                O = np.einsum('ijr,jkr->kir', U, V)
                norm_sq = np.sum(O**2, axis=2, keepdims=True)
                norm_sq[norm_sq < 1e-12] = 1.0
                W = O / norm_sq
                
                # Check convergence every 10 iterations
                if it % 10 == 9:
                    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # U: (n,m,R), V: (m,p,R), W: (p,n,R)
                    # Target: (n, p, m) or similar?
                    # T[i, j, k] where i in 0..n-1, j in 0..m-1, k in 0..p-1
                    # einsum: 'ijr,jkr,kir->ijk'
                    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                    err = np.max(np.abs(T - 1.0))
                    if err < 1e-8:
                        # Success! Convert to list format
                        # U (n, m, R) -> (n*m, R)
                        # Flatten first two dims: row index i*m + j
                        U_list = U.reshape(n*m, R).tolist()
                        # V (m, p, R) -> (m*p, R)
                        # Flatten first two dims: row index j*p + k
                        V_list = V.reshape(m*p, R).tolist()
                        # W (p, n, R) -> (p*n, R)
                        # Flatten first two dims: row index k*n + i
                        W_list = W.reshape(p*n, R).tolist()
                        
                        best_rank = R
                        best_U, best_V, best_W = U_list, V_list, W_list
                        found = True
                        break
                # Early exit if error is increasing significantly? 
                # The heuristic usually converges or diverges.
                # We just rely on the iteration limit and restarts.
                
            if found:
                break
                
        if found:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
