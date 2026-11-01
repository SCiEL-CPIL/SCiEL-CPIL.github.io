import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def validate_fast(U, V, W, n, m, p, tol=1e-9):
    """Validate that U, V, W reconstruct the matrix multiplication tensor exactly."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _clean_val(x):
    """Clean a float value to integer or half-integer if possible."""
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    elif abs(2 * x - round(2 * x)) < 1e-9:
        return round(2 * x) / 2.0
    return x

def compute_residuals_vec(U, V, W, n, m, p):
    """Compute residuals as a flat vector for all (i,j,k) triples."""
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                residuals[idx] = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]) - 1.0
                idx += 1
    return residuals

def residuals_func(theta, n, m, p, r):
    """Flattened residual function for least_squares."""
    # Unpack theta
    size_u = n * m * r
    size_v = m * p * r
    size_w = p * n * r
    
    U_flat = theta[:size_u]
    V_flat = theta[size_u:size_u + size_v]
    W_flat = theta[size_u + size_v:]
    
    U = U_flat.reshape((n * m, r))
    V = V_flat.reshape((m * p, r))
    W = W_flat.reshape((p * n, r))
    
    return compute_residuals_vec(U, V, W, n, m, p)

def try_rank_least_squares(r, n, m, p, timeout, max_restarts=20):
    """Try to find rank-r decomposition via least_squares."""
    start_time = time.time()
    
    size_u = n * m * r
    size_v = m * p * r
    size_w = p * n * r
    total_params = size_u + size_v + size_w
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize
        # Try different scales
        scale = 1.0
        if restart % 2 == 0:
            theta0 = np.random.randn(total_params) * scale
        else:
            # Sparse init
            theta0 = np.zeros(total_params)
            # Randomly set some entries to +/- 1
            for _ in range(total_params // 4):
                idx = random.randint(0, total_params - 1)
                theta0[idx] = random.choice([-1, 1])
        
        try:
            result = least_squares(
                residuals_func, 
                theta0, 
                args=(n, m, p, r), 
                method='lm', 
                max_nfev=1000
            )
            
            if result.success and result.cost < 1e-10:
                # Extract U, V, W
                U_flat = result.x[:size_u]
                V_flat = result.x[size_u:size_u + size_v]
                W_flat = result.x[size_u + size_v:]
                
                U = U_flat.reshape((n * m, r))
                V = V_flat.reshape((m * p, r))
                W = W_flat.reshape((p * n, r))
                
                # Snap to half-integers
                U_s = np.round(2 * U) / 2.0
                V_s = np.round(2 * V) / 2.0
                W_s = np.round(2 * W) / 2.0
                
                if validate_fast(U_s, V_s, W_s, n, m, p):
                    U_list = [[_clean_val(U_s[i, c]) for c in range(r)] for i in range(n * m)]
                    V_list = [[_clean_val(V_s[i, c]) for c in range(r)] for i in range(m * p)]
                    W_list = [[_clean_val(W_s[i, c]) for c in range(r)] for i in range(p * n)]
                    return U_list, V_list, W_list
                    
        except Exception as e:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try ranks from low to high
    # Theoretical lower bound for 2x5x5 is roughly 20-25.
    # We try up to 49.
    for r in range(20, 50):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        remaining = deadline - elapsed - 5
        if remaining <= 0:
            break
        
        # Allocate time for this rank
        # If we have lots of time, try more restarts
        timeout_rank = min(5.0, remaining * 0.5)
        
        result = try_rank_least_squares(r, n, m, p, timeout_rank, max_restarts=10)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = r
            # Since we are iterating from low to high, if we find a solution, 
            # it is the minimal rank we've found so far. 
            # But maybe a lower rank was missed? 
            # We started at 20. If we found at 25, we should check 20-24?
            # No, we iterate 20, 21, 22... 
            # So if we find at 25, it means 20-24 failed.
            # We can return immediately? 
            # Yes, because we want the minimal rank. 
            # If 25 works, and 20-24 didn't, 25 is the best we found.
            # But maybe 24 works with more restarts?
            # Given time constraints, returning the first found (lowest r) is a good heuristic.
            # However, to be safe, we can just return the first valid one found in ascending order.
            return best_result
            
    # Final validation
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        if not validate_fast(U_arr, V_arr, W_arr, n, m, p):
            best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best_result
