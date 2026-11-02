import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_decomposition_error(U, V, W, T):
    """
    Compute the error between the tensor product sum and T.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    # Efficient way: 
    # Sum over r: U[:, r] * V[:, r] * W[:, r]
    # This is a tensor of shape (n*m, m*p, p*n)
    # We can do this by iterating over r, but that's slow in Python loops.
    # Using einsum:
    # np.einsum('ur, vr, wr -> uvw', U, V, W)
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    
    error = np.linalg.norm(reconstructed - T)
    return error

def _local_search(U, V, W, T, values, max_iter=10000, timeout=10):
    """
    Local search to minimize error.
    U, V, W are numpy arrays.
    values is the set of allowed values.
    """
    start_time = time.time()
    current_error = _compute_decomposition_error(U, V, W, T)
    
    # Flatten indices for U, V, W
    n_u, n_v, n_w = U.size, V.size, W.size
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0.0:
            break
            
        # Pick a random variable from U, V, or W
        # Weighted by size? Or just uniform?
        # Let's pick uniformly from all variables
        total_vars = n_u + n_v + n_w
        var_idx = random.randint(0, total_vars - 1)
        
        if var_idx < n_u:
            mat = U
            idx = var_idx
        elif var_idx < n_u + n_v:
            mat = V
            idx = var_idx - n_u
        else:
            mat = W
            idx = var_idx - n_u - n_v
            
        # Pick a new value
        old_val = mat.flat[idx]
        new_val = random.choice(values)
        
        # Try the change
        mat.flat[idx] = new_val
        new_error = _compute_decomposition_error(U, V, W, T)
        
        if new_error < current_error:
            current_error = new_error
        else:
            # Revert
            mat.flat[idx] = old_val
            
    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R.
    """
    start_time = time.time()
    
    # Initial random guess
    # Use small integers/half-integers
    U = np.random.choice(values, size=(n * m, R))
    V = np.random.choice(values, size=(m * p, R))
    W = np.random.choice(values, size=(p * n, R))
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Run several restarts
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        # Random initialization
        U = np.random.choice(values, size=(n * m, R))
        V = np.random.choice(values, size=(m * p, R))
        W = np.random.choice(values, size=(p * n, R))
        
        # Local search
        err = _local_search(U, V, W, T, values, max_iter=5000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error == 0.0:
            break
            
        if time.time() - start_time > time_limit:
            break
            
    if best_error == 0.0:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space
    # Integers and half-integers
    values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    
    # Try ranks lower than schoolbook
    # Known upper bound for 2x3x4 is likely around 18-20.
    # Let's try from 15 up to 23.
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    for R in range(15, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        # Time budget for this rank: remaining time / number of ranks left
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        # Cap time per rank to avoid spending too long on one
        time_budget = min(time_budget, 20.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify again to be safe
            err = _compute_decomposition_error(U, V, W, T)
            if err == 0.0:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                # Found a valid decomposition. 
                # Can we find an even lower one?
                # Yes, continue loop to lower R? 
                # The loop is increasing R. 
                # So if we found R=18, we should check 17, 16...
                # But the loop starts at 15. 
                # So if we find 18, we have already checked 15, 16, 17.
                # So we can break? 
                # No, we want the MINIMAL rank.
                # The loop goes 15, 16, 17...
                # So the first one found is the minimal one in this range.
                # We can break.
                break
                
    # If no better found, return schoolbook
    return best_result
