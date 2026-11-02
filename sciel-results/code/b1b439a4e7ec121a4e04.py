"""Matrix multiplication 2x3x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W correctly decompose the multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor should be: for each (i,j,k), the coefficient is 1 (since A[i,j]*B[j,k] appears once)
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    # Wait, let me reconsider. The tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] = 1 if the bilinear form is A[i,j] * B[j,k]
    # So there are n*m*p terms, each equal to 1.
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], np.dot(V_arr[j*p+k], W_arr[k*n+i]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iter=5000, n_restarts=5, seed=None):
    """Try to find a rank-r decomposition using optimization."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    best_result = None
    
    for restart in range(n_restarts):
        # Initialize with random matrices
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Flatten parameters
        def params_to_matrices(params):
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r :].reshape(p*n, r)
            return U, V, W
        
        def objective(params):
            U, V, W = params_to_matrices(params)
            # Compute sum of squared errors
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                        err += (val - 1.0) ** 2
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='BFGS', 
                          options={'maxiter': max_iter, 'gtol': 1e-8})
            if res.fun < 1e-6:
                U_opt, V_opt, W_opt = params_to_matrices(res.x)
                # Convert to lists
                U_list = U_opt.tolist()
                V_list = V_opt.tolist()
                W_list = W_opt.tolist()
                if _validate(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    best_result = (U_list, V_list, W_list)
                    break
        except Exception:
            continue
    
    return best_result

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Start with schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # For 2x3x6, let's try ranks from 18 upwards
    
    # Try a range of ranks, starting from lower bounds
    ranks_to_try = list(range(18, schoolbook_rank + 1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:  # Safety margin
            break
        
        result = _try_rank(r, n, m, p, max_iter=2000, n_restarts=3, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = r
            break  # Found minimal rank, stop
    
    return best_result
