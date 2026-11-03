import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(n and m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T

def _compute_error_int(U, V, W, T):
    """
    Compute the sum of squared errors using integer arithmetic.
    U, V, W are integer arrays (scaled by 2).
    T is integer array (target values, 1 or 0).
    The reconstruction is sum_r U[:,r] V[:,r] W[:,r].
    If U,V,W are scaled by 2, the result is scaled by 8.
    So we compare (recon/8) vs T, i.e., recon_int vs 8*T.
    """
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target_scaled = 8 * T
    diff = recon - target_scaled
    return int(np.sum(diff * diff))

def _local_search_int(U, V, W, T, values, max_iter=5000, timeout=5.0):
    """
    Discrete local search using integer arithmetic.
    U, V, W are integer arrays.
    values are the allowed integer values (scaled by 2).
    """
    start_time = time.time()
    current_error = _compute_error_int(U, V, W, T)
    
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    shape_U = U.shape
    shape_V = V.shape
    shape_W = W.shape
    
    # Precompute for faster error update
    # For a change in U[u,r] by delta, the error change can be computed quickly
    # But for simplicity, we'll use full recomputation with a tighter loop
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0:
            break
            
        # Pick a random variable
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
            
        old_val = mat.flat[idx]
        
        # Try all possible new values
        best_new_val = old_val
        best_new_error = current_error
        
        for new_val in values:
            if new_val == old_val:
                continue
            mat.flat[idx] = new_val
            new_error = _compute_error_int(U, V, W, T)
            if new_error < best_new_error:
                best_new_error = new_error
                best_new_val = new_val
        
        # Apply the best change
        if best_new_val != old_val:
            mat.flat[idx] = best_new_val
            current_error = best_new_error
        else:
            mat.flat[idx] = old_val

    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R using discrete search.
    """
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Number of restarts based on time
    num_restarts = 30
    time_per_restart = time_limit / num_restarts
    
    int_values = [int(v * 2) for v in values]
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Sparse initialization: mostly zeros with a few non-zero entries
        density = 0.3
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        
        for r in range(R):
            # Each column has some non-zero entries
            num_nonzero_u = max(1, int(n * m * density))
            num_nonzero_v = max(1, int(m * p * density))
            num_nonzero_w = max(1, int(p * n * density))
            
            u_idx = random.sample(range(n * m), min(num_nonzero_u, n * m))
            v_idx = random.sample(range(m * p), min(num_nonzero_v, m * p))
            w_idx = random.sample(range(p * n), min(num_nonzero_w, p * n))
            
            for idx in u_idx:
                U[idx, r] = random.choice(int_values)
            for idx in v_idx:
                V[idx, r] = random.choice(int_values)
            for idx in w_idx:
                W[idx, r] = random.choice(int_values)
        
        # Run local search
        err = _local_search_int(U, V, W, T, int_values, max_iter=2000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error == 0:
        return best_U / 2.0, best_V / 2.0, best_W / 2.0
    return None

def _structured_init(R, n, m, p):
    """
    Create a structured initialization based on the tensor's algebraic structure.
    For 2x3x4, we try to group the schoolbook terms.
    """
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Schoolbook has 24 terms. Try to merge them.
    # For each (i,j,k), the term is at U[i*m+j], V[j*p+k], W[k*n+i]
    # We can try to assign multiple (i,j,k) to the same rank r
    
    # Simple approach: assign terms to ranks in a round-robin fashion
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for idx, (i, j, k) in enumerate(terms):
        r = idx % R
        u = i * m + j
        v = j * p + k
        w = k * n + i
        # Add to the rank r
        U[u, r] += 1
        V[v, r] += 1
        W[w, r] += 1
    
    return U, V, W

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
    
    # Define search space: integers and half-integers
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Try ranks lower than schoolbook
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    for R in range(15, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        # Time budget for this rank
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        time_budget = min(time_budget, 15.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify with original float logic to be safe
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
