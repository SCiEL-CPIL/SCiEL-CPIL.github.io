import time
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Efficient vectorized verification
    # We need to check for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # We can iterate over i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_arr[i * m + j, :]
                v_vec = V_arr[j * p + k, :]
                w_vec = W_arr[k * n + i, :]
                s = np.sum(u_vec * v_vec * w_vec)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _round_to_half(arr):
    """Round array to nearest half-integer."""
    return np.round(2 * arr) / 2.0

def _find_decomposition(r, n=3, m=3, p=3, time_budget=10.0, restarts=5):
    """
    Try to find a rank-r decomposition using random restarts and scipy.optimize.least_squares.
    """
    start = time.time()
    
    # Variable sizes
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    total_vars = size_U + size_V + size_W
    
    def residuals(params):
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:total_vars]
        
        U = U_flat.reshape(n * m, r)
        V = V_flat.reshape(m * p, r)
        W = W_flat.reshape(p * n, r)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i * m + j, :]
                    v_vec = V[j * p + k, :]
                    w_vec = W[k * n + i, :]
                    val = np.sum(u_vec * v_vec * w_vec)
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_err = np.inf
    best_params = None
    
    for restart in range(restarts):
        if time.time() - start > time_budget:
            break
            
        # Random initialization
        # Scale initialization to avoid large gradients
        init = np.random.normal(0, 1.0, size=total_vars)
        
        try:
            # Use 'lm' method which is fast for small problems
            res = optimize.least_squares(residuals, init, method='lm', max_nfev=200)
            err = np.sum(res.fun**2)
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
            if best_err < 1e-12:
                break
        except Exception:
            pass
            
    if best_params is None or best_err > 1e-6:
        return None
        
    # Extract and round to half-integers
    U_flat = best_params[:size_U]
    V_flat = best_params[size_U:size_U + size_V]
    W_flat = best_params[size_U + size_V:total_vars]
    
    U = _round_to_half(U_flat.reshape(n * m, r))
    V = _round_to_half(V_flat.reshape(m * p, r))
    W = _round_to_half(W_flat.reshape(p * n, r))
    
    # Verify the rounded solution
    if _verify(U, V, W, n, m, p):
        return U, V, W
        
    return None

def _get_schoolbook(n=3, m=3, p=3):
    """Construct the standard rank-nmp schoolbook decomposition."""
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

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Build guaranteed schoolbook rank-27 decomposition
    U_school, V_school, W_school = _get_schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # 2. Try to find lower rank decompositions
    # We try ranks from 26 down to 20
    # Allocate time budget based on rank
    ranks_to_try = [26, 25, 24, 23, 22, 21]
    
    for r in ranks_to_try:
        if time.time() - start_time > timeout - 5:
            break
            
        # Give more time to higher ranks (closer to 27) as they are more likely to have solutions
        # If rank 26 doesn't work, rank 25 is less likely, etc.
        # But we must try.
        budget = 15.0 if r >= 25 else 10.0
        restarts = 10 if r >= 25 else 5
        
        result = _find_decomposition(r, n, m, p, time_budget=budget, restarts=restarts)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found a valid lower rank, stop
            
    return best_result
# EVOLVE-BLOCK-END
