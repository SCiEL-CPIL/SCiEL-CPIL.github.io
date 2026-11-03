import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    # Target tensor
    T = np.ones((n, m, p))
    
    # Heuristic search for lower rank
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from 62 down to 50
    for target_rank in range(rank_sb - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for _ in range(5):
            if time.time() > deadline:
                break
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Check if this decomposition is exact?
            # This is hard to check directly without solving.
            # We can compute the tensor product and compare to T.
            # Reconstruct T from U, V, W
            # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # This is O(nmp * rank)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-9):
                # Found a valid decomposition!
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
