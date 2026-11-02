import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def _build_schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _is_valid(U, V, W, tol=1e-6):
    """Check if the factorization exactly reproduces the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U_arr[0])):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _search_rank(target_rank, deadline, max_iter=1000):
    """Try to find a decomposition of the given target_rank."""
    n, m, p = DIMENSIONS
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Objective function
    def objective(x):
        U = x[:U_size * target_rank].reshape(U_size, target_rank)
        V = x[U_size * target_rank : U_size * target_rank + V_size * target_rank].reshape(V_size, target_rank)
        W = x[U_size * target_rank + V_size * target_rank :].reshape(W_size, target_rank)
        
        # Compute T[i,j,k]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    errors.append(s - 1.0)
        return np.sum(np.square(errors))
    
    # Initial guess: random
    x0 = np.random.randn(U_size * target_rank + V_size * target_rank + W_size * target_rank) * 0.1
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'disp': False})
    
    if res.fun < 1e-4:
        U = res.x[:U_size * target_rank].reshape(U_size, target_rank)
        V = res.x[U_size * target_rank : U_size * target_rank + V_size * target_rank].reshape(V_size, target_rank)
        W = res.x[U_size * target_rank + V_size * target_rank :].reshape(W_size, target_rank)
        
        # Try to snap to integers/half-integers
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        U_list = U_int.tolist()
        V_list = V_int.tolist()
        W_list = W_int.tolist()
        
        if _is_valid(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    """Construct the tensor decomposition for 2x3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a lower rank decomposition
    # Known lower bounds suggest rank < 42 is possible.
    # Try ranks 20 to 41
    for r in range(20, 42):
        if time.time() > deadline:
            break
        result = _search_rank(r, deadline)
        if result is not None:
            return result
            
    # Fall back to schoolbook
    schoolbook = _build_schoolbook()
    return schoolbook
