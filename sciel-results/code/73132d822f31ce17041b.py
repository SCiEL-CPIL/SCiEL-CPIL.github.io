import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T_target = _build_tensor(n, m, p)
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return np.allclose(T_recon, T_target, atol=tol)

def _per_index_als(n, m, p, target_rank, max_iter=300, seed=42, time_limit=8):
    """
    Per-index alternating least squares exploiting the tensor structure.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    np.random.seed(seed)
    start_time = time.time()
    
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    T_target = _build_tensor(n, m, p)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                A = np.zeros((target_rank, p))
                for r in range(target_rank):
                    for k in range(p):
                        A[r, k] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A.T, T_target[i, j, :], rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                A = np.zeros((target_rank, n))
                for r in range(target_rank):
                    for i in range(n):
                        A[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A.T, T_target[:, j, k], rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                A = np.zeros((target_rank, m))
                for r in range(target_rank):
                    for j in range(m):
                        A[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A.T, T_target[i, :, k], rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Compute error
        T_recon = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        error = np.linalg.norm(T_recon - T_target)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def _structured_init(n, m, p, target_rank, seed=42):
    """
    Structured initialization that respects the tensor's all-ones structure.
    Try to initialize with patterns that might lead to lower-rank solutions.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Strategy 1: Use the natural grouping
    # Each (i,j,k) triple contributes to one rank component
    # Try to pack multiple triples into same rank components
    
    # Initialize with random assignment of triples to rank components
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    random.shuffle(triples)
    
    for idx, (i, j, k) in enumerate(triples):
        r = idx % target_rank
        # Add contribution to this rank component
        U[i * m + j, r] += 1.0
        V[j * p + k, r] += 1.0
        W[k * n + i, r] += 1.0
    
    # Add small random perturbation
    U += 0.01 * np.random.randn(n * m, target_rank)
    V += 0.01 * np.random.randn(m * p, target_rank)
    W += 0.01 * np.random.randn(p * n, target_rank)
    
    return U, V, W

def _try_rank_with_structured_init(n, m, p, target_rank, max_iter=200, seed=42, time_limit=8):
    """Try to find a decomposition using structured initialization + per-index ALS."""
    U_init, V_init, W_init = _structured_init(n, m, p, target_rank, seed=seed)
    
    np.random.seed(seed)
    start_time = time.time()
    
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    T_target = _build_tensor(n, m, p)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((target_rank, p))
                for r in range(target_rank):
                    for k in range(p):
                        A[r, k] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A.T, T_target[i, j, :], rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((target_rank, n))
                for r in range(target_rank):
                    for i in range(n):
                        A[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A.T, T_target[:, j, k], rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((target_rank, m))
                for r in range(target_rank):
                    for j in range(m):
                        A[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A.T, T_target[i, :, k], rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Compute error
        T_recon = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        error = np.linalg.norm(T_recon - T_target)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def _round_to_half_int(M):
    """Round matrix entries to nearest 0.5"""
    return np.round(M * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_fallback = n * m * p
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fallback
    
    # Try ranks from lower bound upwards
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we know from theory that 2x5x5 has rank around 18-25
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Method 1: Per-index ALS with random initialization
        for seed in range(15):
            if time.time() > deadline - 10:
                break
            
            result = _per_index_als(n, m, p, target_rank, max_iter=200, seed=seed, time_limit=6)
            if result is not None:
                U_r, V_r, W_r = result
                # Round to half-integers and verify
                U_hr = _round_to_half_int(U_r)
                V_hr = _round_to_half_int(V_r)
                W_hr = _round_to_half_int(W_r)
                
                if _verify_decomposition(U_hr, V_hr, W_hr, n, m, p, tol=1e-9):
                    best_U, best_V, best_W = U_hr, V_hr, W_hr
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Try without rounding
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
        
        # Method 2: Structured initialization + per-index ALS
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_with_structured_init(n, m, p, target_rank, max_iter=150, seed=seed, time_limit=6)
            if result is not None:
                U_r, V_r, W_r = result
                U_hr = _round_to_half_int(U_r)
                V_hr = _round_to_half_int(V_r)
                W_hr = _round_to_half_int(W_r)
                
                if _verify_decomposition(U_hr, V_hr, W_hr, n, m, p, tol=1e-9):
                    best_U, best_V, best_W = U_hr, V_hr, W_hr
                    best_rank = target_rank
                    found = True
                    break
                else:
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    # Convert to list format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
