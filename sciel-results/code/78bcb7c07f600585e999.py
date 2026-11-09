"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_error_vectorized(U, V, W):
    """Compute the max error using vectorized operations."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    # Build index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    # For each (i,j,k), compute sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
    U_sel = U_np[idx_u, :]
    V_sel = V_np[idx_v, :]
    W_sel = W_np[idx_w, :]
    
    prod = U_sel * V_sel * W_sel
    vals = prod.sum(axis=1)
    
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _solve_rank(target_rank, deadline, num_restarts=5):
    """Try to find a rank decomposition using scipy.optimize.least_squares."""
    n, m, p = DIMENSIONS
    N = n * m * p
    
    # Precompute index arrays for flattening
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    def residuals(x):
        # x contains flattened U, V, W
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u + size_v]
        W_flat = x[size_u + size_v:]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        
        vals = (U_sel * V_sel * W_sel).sum(axis=1)
        return vals - 1.0
    
    best_err = float('inf')
    best_x = None
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(restart * 1000 + target_rank)
        
        # Random initialization
        # Scale to avoid large gradients initially
        scale = 0.5
        x0 = np.random.randn(n * m * target_rank + m * p * target_rank + p * n * target_rank) * scale
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            err = np.max(np.abs(result.fun))
            
            if err < best_err:
                best_err = err
                best_x = result.x
                
            if best_err < 1e-8:
                break
        except Exception:
            pass
            
    if best_x is None:
        return None
        
    if best_err < 1e-6:
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        
        U_flat = best_x[:size_u]
        V_flat = best_x[size_u:size_u + size_v]
        W_flat = best_x[size_u + size_v:]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        err_r = _compute_error_vectorized(U_r, V_r, W_r)
        if err_r < 1e-10:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Start from a reasonable guess, e.g., 100, down to 60
    # If we find one, we can try lower
    for target_rank in range(100, 59, -1):
        if time.time() > deadline - 20:
            break
            
        result = _solve_rank(target_rank, deadline, num_restarts=3)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Do not break immediately, try to go lower if time permits
            # But if we found a very low rank, we might stop
            if target_rank < 70:
                break

    return best_result
# EVOLVE-BLOCK-END
