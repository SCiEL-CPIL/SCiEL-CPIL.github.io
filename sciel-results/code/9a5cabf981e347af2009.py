import time
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _round_to_half(x):
    return np.round(2 * x) / 2.0

def _find_decomposition(r, n=3, m=3, p=3, time_budget=10.0, restarts=5):
    """
    Try to find a rank-r decomposition using random restarts and scipy.optimize.least_squares.
    """
    start = time.time()
    U_shape = (n * m, r)
    V_shape = (m * p, r)
    W_shape = (p * n, r)
    
    # Total variables: 3 * 9 * r
    # Residuals: 27
    
    def residuals(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:2*n*m*r]
        W_flat = params[2*n*m*r:3*n*m*r]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_err = np.inf
    best_params = None
    
    for restart in range(restarts):
        if time.time() - start > time_budget:
            break
            
        # Random initialization
        # Scale matters. Let's try standard normal.
        init = np.random.normal(0, 1, size=3 * n * m * r)
        
        try:
            res = optimize.least_squares(residuals, init, method='lm', max_nfev=100)
            err = np.sum(res.fun**2)
            if err < best_err:
                best_err = err
                best_params = res.x
                
            if best_err < 1e-8:
                break
        except Exception:
            pass
            
    if best_params is None or best_err > 1e-6:
        return None
        
    # Extract and round
    U_flat = best_params[:n*m*r]
    V_flat = best_params[n*m*r:2*n*m*r]
    W_flat = best_params[2*n*m*r:3*n*m*r]
    
    U = _round_to_half(U_flat.reshape(n*m, r))
    V = _round_to_half(V_flat.reshape(m*p, r))
    W = _round_to_half(W_flat.reshape(p*n, r))
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return U, V, W
        
    return None

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed schoolbook rank-27 decomposition
    rank_schoolbook = n * m * p  # 27
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # We try ranks 23, 24, 25, 26
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > timeout - 5:
            break
            
        # Allocate time based on rank difficulty? 
        # Let's give 10s to rank 23, 5s to 24, 5s to 25, 5s to 26
        budget = 10.0 if r == 23 else 5.0
        restarts = 5 if r == 23 else 3
        
        result = _find_decomposition(r, n, m, p, time_budget=budget, restarts=restarts)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break # Found a lower rank, stop searching for higher ranks? 
                   # Actually, we want the MINIMAL rank. 
                   # Since we iterate 23, 24, 25, 26, the first one found is the lowest.
                   # So we can break.
            
    return best_result
# EVOLVE-BLOCK-END
