import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > tol:
                    return False
    return True

def _is_half_integer(arr, tol=1e-9):
    """Check if all entries are integers or half-integers."""
    doubled = 2 * np.array(arr, dtype=np.float64)
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _round_to_half(x):
    return np.round(2 * np.array(x, dtype=np.float64)) / 2.0

def _polish_decomposition(U, V, W, n=3, m=3, p=3, max_iter=200, tol=1e-12):
    """
    Polish a near-exact decomposition using iterative refinement.
    Tries to drive the residual to zero while maintaining structure.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    for iteration in range(max_iter):
        # Compute current residual
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    res = abs(s - 1.0)
                    max_res = max(max_res, res)
        
        if max_res < tol:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k] * W[k*n+i]
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j] = U_new
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j] * W[k*n+i]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k] = V_new
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j] * V[j*p+k]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i] = W_new
                except:
                    pass
    
    return U, V, W

def _try_rank_r(r, n=3, m=3, p=3, time_budget=10.0, restarts=10):
    """
    Try to find a valid rank-r decomposition using structured initialization
    and polishing.
    """
    start = time.time()
    
    # Strategy: Use block-structured initializations inspired by Strassen
    # For 3x3, we can try various structured patterns
    
    for restart in range(restarts):
        if time.time() - start > time_budget:
            break
        
        np.random.seed(42 + r * 100 + restart)
        
        # Initialize with random values scaled appropriately
        U = np.random.randn(n * m, r) * 0.8
        V = np.random.randn(m * p, r) * 0.8
        W = np.random.randn(p * n, r) * 0.8
        
        # Run polishing
        U_pol, V_pol, W_pol = _polish_decomposition(U, V, W, n, m, p, 
                                                      max_iter=100, tol=1e-10)
        
        # Check if close to exact
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_pol[i*m+j] * V_pol[j*p+k] * W_pol[k*n+i])
                    max_res = max(max_res, abs(s - 1.0))
        
        if max_res < 1e-8:
            # Round to half-integers and verify
            U_r = _round_to_half(U_pol)
            V_r = _round_to_half(V_pol)
            W_r = _round_to_half(W_pol)
            
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    
    return None

def _schoolbook_decomposition(n=3, m=3, p=3):
    """Return the guaranteed rank-27 schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p
    
    # Try ranks from 23 upward (known lower bound for 3x3 is 19, 
    # but 23 is a practical target with known constructions)
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > timeout - 10:
            break
        
        # Allocate more time for lower ranks (harder to find)
        budget = min(25.0, (timeout - 10 - (time.time() - start_time)) * 0.8)
        restarts = 15 if r == 23 else 10
        
        result = _try_rank_r(r, n, m, p, time_budget=budget, restarts=restarts)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = r
            break  # Found a valid lower-rank decomposition, stop
    
    return best_result
# EVOLVE-BLOCK-END
