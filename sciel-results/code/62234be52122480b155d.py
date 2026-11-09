import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                s = np.dot(np.multiply(u_val, v_val), w_val)
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _random_decomposition(n, m, p, rank, rng):
    """Generate a random decomposition of given rank."""
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U, V, W

def _objective(U, V, W, n, m, p):
    """Compute the objective: sum of squared errors over all (i,j,k) entries."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                s = np.dot(np.multiply(u_val, v_val), w_val)
                total_error += (s - 1.0) ** 2
    return total_error

def _try_optimize(n, m, p, rank, rng, time_budget):
    """Try to find a decomposition of given rank using random restarts and gradient descent."""
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    max_restarts = 30
    max_iters = 100
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        U, V, W = _random_decomposition(n, m, p, rank, rng)
        current_error = _objective(U, V, W, n, m, p)
        
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute gradients and update
            # For efficiency, precompute some terms
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j]
                        v_val = V[j * p + k]
                        w_val = W[k * n + i]
                        s = np.dot(np.multiply(u_val, v_val), w_val)
                        err = s - 1.0
                        
                        # d(error)/d(U[i*m+j, r]) = 2 * err * V[j*p+k, r] * W[k*n+i, r]
                        grad_U[i * m + j] += 2 * err * np.multiply(v_val, w_val)
                        grad_V[j * p + k] += 2 * err * np.multiply(u_val, w_val)
                        grad_W[k * n + i] += 2 * err * np.multiply(u_val, v_val)
            
            # Update with learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            new_error = _objective(U, V, W, n, m, p)
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if new_error > current_error:
                lr *= 0.5
            else:
                lr = min(lr * 1.1, 0.1)
            
            current_error = new_error
            
            if best_error < 1e-8:
                break
    
    if best_U is not None and best_error < 1e-6:
        # Convert to list of lists with rounded values
        U_list = [[round(float(x), 10) for x in row] for row in best_U]
        V_list = [[round(float(x), 10) for x in row] for row in best_V]
        W_list = [[round(float(x), 10) for x in row] for row in best_W]
        return U_list, V_list, W_list
    
    return None

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook_decomp(n, m, p)
    
    # Try to find lower rank decompositions
    rng = np.random.RandomState(42)
    
    # Try ranks from low to high, stopping when we find a valid decomposition
    # Schoolbook rank is 90. Try to find something lower.
    ranks_to_try = list(range(70, 90))  # Try ranks 70-89
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        time_budget = min(10.0, deadline - time.time())
        if time_budget < 2.0:
            break
        
        result = _try_optimize(n, m, p, rank, rng, time_budget)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
