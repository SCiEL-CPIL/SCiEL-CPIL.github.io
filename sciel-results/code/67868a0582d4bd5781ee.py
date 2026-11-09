import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(W[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the tensor decomposition."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The term is sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
                # We want this to be 1.
                # Vectorized:
                # u_vec = U[i*m+j, :]
                # v_vec = V[j*p+k, :]
                # w_vec = W[k*n+i, :]
                # val = np.dot(u_vec, v_vec * w_vec)
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                error += (val - 1.0)**2
    return error

def _local_search(U, V, W, n, m, p, max_iters=5000, temp=1.0, cooling=0.999):
    """
    Simulated Annealing to find a zero-error solution.
    U, V, W are numpy arrays.
    """
    r = W.shape[1]
    current_error = _compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    current_temp = temp
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # Pick a random matrix and a random entry
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            arr = U
            rows = n * m
        elif mat_choice == 1:
            arr = V
            rows = m * p
        else:
            arr = W
            rows = p * n
            
        idx = np.random.randint(0, rows)
        col = np.random.randint(0, r)
        
        old_val = arr[idx, col]
        # Perturb by a small amount
        # We try integers and half-integers
        delta = np.random.choice([-1, -0.5, 0.5, 1])
        new_val = old_val + delta
        
        arr[idx, col] = new_val
        
        new_error = _compute_error(U, V, W, n, m, p)
        
        # Acceptance criterion
        delta_error = new_error - current_error
        if delta_error < 0 or (current_temp > 0 and np.random.random() < np.exp(-delta_error / current_temp)):
            current_error = new_error
            if current_error < best_error:
                best_error = current_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        else:
            arr[idx, col] = old_val
            
        current_temp *= cooling
        
        if best_error < 1e-8:
            break
            
    return best_U, best_V, best_W, best_error

def _try_rank(r, n, m, p, time_limit=5.0, seed=0):
    """
    Try to find a decomposition of rank r.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() - start > time_limit:
            break
            
        # Initialize with small random integers
        # Range [-2, 2]
        U = rng.randint(-2, 3, size=(n*m, r)).astype(float)
        V = rng.randint(-2, 3, size=(m*p, r)).astype(float)
        W = rng.randint(-2, 3, size=(p*n, r)).astype(float)
        
        # Local search
        U, V, W, err = _local_search(U, V, W, n, m, p, max_iters=1000, temp=1.0, cooling=0.999)
        
        if err < 1e-8:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to reduce rank
    # Lower bound is max(n*m, m*p, p*n) = 20
    # Let's try from best_rank - 1 down to 20
    # We give more time to lower ranks? No, we give a fixed time per rank attempt.
    
    for r in range(best_rank - 1, 19, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_per_rank = min(2.0, (DEADLINE - 20 - (time.time() - start_time)) / 5.0)
        if time_per_rank < 0.5:
            break
            
        result = _try_rank(r, n, m, p, time_limit=time_per_rank, seed=r*17)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = r
            # If we found a lower rank, we can continue to search for even lower
            # But we must be careful with time.
            
    return best
