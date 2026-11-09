import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and check it equals the canonical multiplication tensor (which is 1 for all i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _residuals(x, n, m, p, rank):
    """
    Compute residuals for least_squares.
    x is a flat array of length (n*m + m*p + p*n) * rank.
    Residuals are the differences between the computed tensor entries and 1.
    """
    U_size = n * m * rank
    V_size = m * p * rank
    
    U_flat = x[:U_size]
    V_flat = x[U_size:U_size + V_size]
    W_flat = x[U_size + V_size:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                residuals[idx] = val - 1.0
                idx += 1
    return residuals

def _try_ranked_decomposition(n, m, p, target_rank, max_attempts=10, timeout=5.0):
    """
    Try to find an exact decomposition of rank target_rank using least_squares.
    """
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        x0 = np.random.randn((n * m + m * p + p * n) * target_rank)
        
        try:
            result = least_squares(
                _residuals, 
                x0, 
                args=(n, m, p, target_rank),
                method='lm', 
                max_nfev=200
            )
            
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                x_opt = result.x
                U_size = n * m * target_rank
                V_size = m * p * target_rank
                
                U = x_opt[:U_size].reshape(n * m, target_rank)
                V = x_opt[U_size:U_size + V_size].reshape(m * p, target_rank)
                W = x_opt[U_size + V_size:].reshape(p * n, target_rank)
                
                U_list = [[float(val) for val in row] for row in U]
                V_list = [[float(val) for val in row] for row in V]
                W_list = [[float(val) for val in row] for row in W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
                    
        except Exception:
            continue
            
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    schoolbook = _build_schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    for target_rank in range(50, best_rank):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        attempts = max(2, int(time_left / 5.0))
        
        result = _try_ranked_decomposition(n, m, p, target_rank, 
                                           max_attempts=attempts, 
                                           timeout=min(5.0, time_left - 1.0))
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break

    return best_result
# EVOLVE-BLOCK-END
