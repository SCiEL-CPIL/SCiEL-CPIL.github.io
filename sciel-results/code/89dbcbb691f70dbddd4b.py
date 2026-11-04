"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Check time - if running low, return schoolbook
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    # Verify function
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_true = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m + j, j*p + k, k*n + i] = 1.0
        
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_recon, T_true, atol=1e-10):
            return False
        return True
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: Build decomposition column by column, trying to cover multiple (i,j,k) triples
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(nm, mp, pn) = 32 for the dimensions, but actually
    # the rank must be at least the "slice rank" which for matrix mult is at least max(n,m,p)
    # More refined: rank >= n*p = 24 (fixing j), rank >= m*p = 32 (fixing i), rank >= n*m = 12 (fixing k)
    # So lower bound is 32
    
    lb = max(nm, mp, pn)  # 32
    
    # Try to find a decomposition by iteratively adding columns
    # Each column c has U_col (nm-dim), V_col (mp-dim), W_col (pn-dim)
    # The contribution of column c to the tensor is outer(U_col, V_col, W_col)
    
    # Greedy approach: start with zero tensor, add columns that best reduce residual
    # But this is expensive. Let's try a more structured approach.
    
    # Alternative: try to find a decomposition where columns are "pure" (each covers one (i,j,k))
    # but with some sharing. 
    
    # Let's try: for each j, we need to decompose the n×p sub-tensor for that j.
    # The sub-tensor for fixed j has T[i*m+j, j*p+k, k*n+i] = 1 for all i,k.
    # This is a rank n*p = 24 tensor (schoolbook for that slice).
    
    # But we can potentially share columns across different j values if the patterns align.
    
    # Let's try a systematic search for rank r from lb to schoolbook rank
    # For each r, try to find U, V, W using a local search approach
    
    # First, try a few specific candidate ranks with a heuristic search
    candidate_ranks = list(range(lb, best_rank + 1, 4))
    candidate_ranks.append(best_rank)
    candidate_ranks = sorted(set(candidate_ranks))
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        if r >= best_rank:
            break
        
        # Try to find a rank-r decomposition
        # Use a simple approach: initialize with random values and do gradient descent
        # on the objective: sum of squares of (T_recon - T_true)^2
        
        found = False
        for seed in range(3):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            rng = np.random.RandomState(seed)
            
            # Initialize with small random values
            U = rng.randn(nm, r) * 0.1
            V = rng.randn(mp, r) * 0.1
            W = rng.randn(pn, r) * 0.1
            
            # Build true tensor
            T_true = np.zeros((nm, mp, pn), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_true[i*m + j, j*p + k, k*n + i] = 1.0
            
            # Gradient descent
            eta = 0.005
            for it in range(2000):
                if time.time() - START_TIME > TIME_LIMIT - 15:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T_true - T_recon
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = np.einsum('abc,ar,br->cr', R, U, V)
                
                U += eta * grad_U
                V += eta * grad_V
                W += eta * grad_W
                
                # Check convergence
                if it % 200 == 0:
                    err = np.max(np.abs(T_recon - T_true))
                    if err < 1e-8:
                        # Try to snap to integer/half-integer values
                        for scale in [1, 2]:
                            U_r = np.round(U * scale) / scale
                            V_r = np.round(V * scale) / scale
                            W_r = np.round(W * scale) / scale
                            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                                found = True
                                break
                        if found:
                            break
            
            if found:
                break
        
        if found:
            U_final = np.round(U * 2) / 2
            V_final = np.round(V * 2) / 2
            W_final = np.round(W * 2) / 2
            if verify(U_final.tolist(), V_final.tolist(), W_final.tolist()):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
