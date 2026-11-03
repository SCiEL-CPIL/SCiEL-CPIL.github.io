import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Fast verification using vectorized tensor contraction.
    """
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False

    if np.allclose(T, 1.0, atol=1e-9):
        return True
    return False

def _residual_func(x, n, m, p, R):
    """
    Residual for least_squares.
    x is flat array of length R * (n*m + m*p + p*n)
    """
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    U = x[:R * size_U].reshape(size_U, R)
    V = x[R * size_U : R * size_U + R * size_V].reshape(size_V, R)
    W = x[R * size_U + R * size_V :].reshape(size_W, R)
    
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Residual is T - 1
    return (T - 1.0).flatten()

def _try_find_decomposition(n, m, p, R, time_limit, seed):
    """
    Try to find a rank-R decomposition using least_squares.
    """
    size_U = n * m
    size_V = m * p
    size_W = p * n
    total_vars = R * (size_U + size_V + size_W)
    
    np.random.seed(seed)
    
    # Initial guess: Random small values
    x0 = np.random.randn(total_vars) * 0.1
    
    # We can also try a structured initial guess
    # e.g. U, V, W with some 1s and 0s
    
    start_time = time.time()
    
    try:
        # Use least_squares with 'trf' method
        # x_scale='jac' can help
        result = least_squares(
            _residual_func, 
            x0, 
            args=(n, m, p, R),
            method='trf',
            x_scale='jac',
            max_nfev=1000, # Limit function evaluations
            tol=1e-8
        )
        
        if time.time() - start_time > time_limit:
            return None
            
        # Check if the solution is close to zero residual
        if result.cost < 1e-10:
            # Extract U, V, W
            U = result.x[:R * size_U].reshape(size_U, R)
            V = result.x[R * size_U : R * size_U + R * size_V].reshape(size_V, R)
            W = result.x[R * size_U + R * size_V :].reshape(size_W, R)
            
            # Try rounding to integers or half-integers
            # The problem allows integer/half-integer entries.
            # Let's try rounding to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify_decomposition_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
            
            # Try rounding to integers
            U_rounded_int = np.round(U)
            V_rounded_int = np.round(V)
            W_rounded_int = np.round(W)
            
            if _verify_decomposition_fast(U_rounded_int, V_rounded_int, W_rounded_int, n, m, p):
                return U_rounded_int, V_rounded_int, W_rounded_int
                
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # We want to find the minimal rank.
    # Start from a low rank and go up.
    # Lower bound is roughly max(n,m,p) = 7.
    # Let's try ranks from 10 to 40.
    
    best_result = None
    
    # Try to find a decomposition for increasing ranks
    for R in range(10, 45):
        if time.time() > deadline - 10:
            break
            
        # Try a few random seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            res = _try_find_decomposition(n, m, p, R, deadline - time.time() - 5, seed)
            if res is not None:
                best_result = res
                break # Found a valid decomposition for this R, try lower R next? 
                       # No, we are going up. We want the MINIMAL rank.
                       # So we should start from low R and stop at the first valid one.
                       
        if best_result is not None:
            break
            
    if best_result is None:
        # Fallback to schoolbook if search fails
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
        best_result = (U, V, W)
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
