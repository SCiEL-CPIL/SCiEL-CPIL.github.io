import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _solve_rank(n, m, p, R, time_limit=10.0):
    """Try to find a decomposition of rank R using least_squares."""
    start_time = time.time()
    num_vars = R * (n * m + m * p + p * n)
    
    # Define the residual function
    def residual(params):
        U = params[:R * n * m].reshape(n * m, R)
        V = params[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
        W = params[R * (n * m + m * p):].reshape(p * n, R)
        
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
        
        I, J, JP, K, KP, IP = np.meshgrid(
            range(n), range(m), range(m), range(p), range(p), range(n),
            indexing='ij'
        )
        target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
        
        return (T - target).flatten()

    # Try a few random initializations
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small integers
        params = np.random.randint(-2, 3, size=num_vars)
        
        try:
            result = least_squares(residual, params, method='lm', max_nfev=1000)
            if result.success and np.allclose(result.fun, 0, atol=1e-8):
                U = result.x[:R * n * m].reshape(n * m, R)
                V = result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
                W = result.x[R * (n * m + m * p):].reshape(p * n, R)
                # Verify
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 24
    # We try ranks from 47 down to 40 (limited search)
    for target_rank in range(47, 39, -1):
        if time.time() > DEADLINE - 10:
            break
        
        result = _solve_rank(n, m, p, target_rank, time_limit=10.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break # Found a solution, return it
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
