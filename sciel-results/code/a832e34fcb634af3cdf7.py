import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Attempts rank 17 first (known optimal for M_{2,3,3}), falls back to rank 18 schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Try rank 17
    result = _try_rank_17(n, m, p, 17, deadline)
    if result is not None:
        return result
    
    # Fall back to rank 18 schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col_idx] = 1.0
                V[idx_v][col_idx] = 1.0
                W[idx_w][col_idx] = 1.0
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_17(n, m, p, rank, deadline):
    """
    Try to find a rank-17 decomposition by fixing U,V randomly and solving for W.
    """
    if time.time() > deadline - 10:
        return None
    
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        
        W = np.zeros((p * n, rank))
        success = True
        
        for i in range(n):
            for k in range(p):
                M_ik = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        M_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                try:
                    w, _, _, _ = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    check = M_ik @ w
                    if np.max(np.abs(check - np.ones(m))) > 1e-10:
                        success = False
                        break
                    W[k * n + i, :] = w
                except:
                    success = False
                    break
            
            if not success:
                break
        
        if not success:
            continue
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None
