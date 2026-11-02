import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _als_decompose(n, m, p, rank, max_iters=100, seed=42):
    """
    Use ALS to find a rank `rank` decomposition.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Precompute indices for efficiency? 
    # The structure is dense in the sense that for each (i,j) we sum over k.
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                # Construct A_ij: p x rank
                # A_ij[k, r] = V[j*p + k, r] * W[k*n + i, r]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                # Target is vector of ones of length p
                t = np.ones(p)
                # Solve A u = t
                try:
                    u, residuals, rank_l, s = np.linalg.lstsq(A, t, rcond=None)
                    U[i*m + j, :] = u
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # Construct B_jk: n x rank
                # B_jk[i, r] = U[i*m + j, r] * W[k*n + i, r]
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m + j, :] * W[k*n + i, :]
                # Target is vector of ones of length n
                t = np.ones(n)
                try:
                    v, residuals, rank_l, s = np.linalg.lstsq(B, t, rcond=None)
                    V[j*p + k, :] = v
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Construct C_ki: m x rank
                # C_ki[j, r] = U[i*m + j, r] * V[j*p + k, r]
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m + j, :] * V[j*p + k, :]
                # Target is vector of ones of length m
                t = np.ones(m)
                try:
                    w, residuals, rank_l, s = np.linalg.lstsq(C, t, rcond=None)
                    W[k*n + i, :] = w
                except:
                    pass
        
        # Check convergence
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    residual += (s - 1.0)**2
        
        if residual < 1e-12:
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try ranks from 99 down to 90
    for target_rank in range(best_rank - 1, max(50, best_rank - 10), -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few seeds
        for seed in [42, 123, 456, 789, 101]:
            if time.time() > deadline:
                break
            
            U, V, W = _als_decompose(n, m, p, target_rank, max_iters=50, seed=seed)
            
            if U is not None:
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower? Or just return?
            # We can try to go lower if time permits
            pass
            
    return best_result
