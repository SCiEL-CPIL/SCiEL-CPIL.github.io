import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    R = len(U[0])
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sum_val = 0.0
                for r in range(R):
                    sum_val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(sum_val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(n, m, p, target_rank, timeout=10):
    """
    Try to find a decomposition of rank target_rank.
    Returns dict or None.
    """
    start_time = time.time()
    # Equations: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # Variables: U, V, W entries.
    
    # We can fix one matrix to simplify? No, symmetry is broken.
    # Let's use a random initialization and optimize.
    
    num_vars = target_rank * (n * m + m * p + p * p) # Wait, W is p*n
    num_vars = target_rank * (n * m + m * p + p * n)
    
    # Residual function for least squares
    def residuals(x):
        U = np.array(x[0:target_rank * n * m]).reshape(n * m, target_rank)
        V = np.array(x[target_rank * n * m: target_rank * (n * m + m * p)]).reshape(m * p, target_rank)
        W = np.array(x[target_rank * (n * m + m * p):]).reshape(p * n, target_rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    res.append(val - 1.0)
        return np.array(res)

    best_err = float('inf')
    best_x = None
    
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        x0 = np.random.randn(num_vars) * 0.1
        # Try to optimize
        try:
            from scipy.optimize import least_squares
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.cost < best_err:
                best_err = res.cost
                best_x = res.x
                if best_err < 1e-10:
                    break
        except Exception:
            pass

    if best_x is None or best_err > 1e-6:
        return None

    # Try to round to half-integers
    U = np.array(best_x[0:target_rank * n * m]).reshape(n * m, target_rank)
    V = np.array(best_x[target_rank * n * m: target_rank * (n * m + m * p)]).reshape(m * p, target_rank)
    W = np.array(best_x[target_rank * (n * m + m * p):]).reshape(p * n, target_rank)
    
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Check exactness
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    
    if _validate(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
        
    # If rounding failed, check if the floating point solution is "close enough" 
    # and maybe we can just return the floats? 
    # The task says "integer/half-integer entries". So we MUST round.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110 # Safety margin
    
    # Try ranks from 42 down to 40
    for rank in range(42, 39, -1):
        if time.time() > deadline:
            break
        result = _try_find_decomposition(n, m, p, rank, timeout=10)
        if result:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
