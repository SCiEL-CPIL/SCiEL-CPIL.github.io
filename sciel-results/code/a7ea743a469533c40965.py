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
    try:
        reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return float('inf')
    
    error = np.linalg.norm(reconstructed - T)
    return error

def _local_search(U, V, W, T, values, max_iter=20000, timeout=10):
    """
    Local search to minimize error.
    U, V, W are numpy arrays.
    values is the set of allowed values.
    """
    start_time = time.time()
    current_error = _compute_decomposition_error(U, V, W, T)
    
    # Flatten indices for U, V, W
    n_u, n_v, n_w = U.size, V.size, W.size
    
    # Pre-compute slices for faster access? No, flat index is fast enough.
    # We can speed up error calculation by only checking the difference?
    # No, we need the full tensor difference for L-BFGS, but for local search
    # we just need the norm.
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error < 1e-10:
            break
            
        # Pick a random variable from U, V, or W
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
        
        if new_val == old_val:
            continue

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
    Uses gradient descent on real values followed by rounding and local search.
    """
    start_time = time.time()
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # We will try a few random restarts with gradient descent
    # Increase restarts to improve chance of finding a good basin
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Random Real Initialization
        # Scale matters. 1.0 is often better than 0.5 for these problems.
        U_real = np.random.randn(n * m, R)
        V_real = np.random.randn(m * p, R)
        W_real = np.random.randn(p * n, R)
        
        # Flatten for optimization
        U_flat = U_real.flatten()
        V_flat = V_real.flatten()
        W_flat = W_real.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        # Objective function
        def objective(x):
            U_r = x[:U_flat.size].reshape(n * m, R)
            V_r = x[U_flat.size:U_flat.size + V_flat.size].reshape(m * p, R)
            W_r = x[U_flat.size + V_flat.size:].reshape(p * n, R)
            
            # Compute error
            try:
                recon = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            except Exception:
                return 1e10
            err = np.sum((recon - T) ** 2)
            return err
        
        # Gradient function
        def gradient(x):
            U_r = x[:U_flat.size].reshape(n * m, R)
            V_r = x[U_flat.size:U_flat.size + V_flat.size].reshape(m * p, R)
            W_r = x[U_flat.size + V_flat.size:].reshape(p * n, R)
            
            recon = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            err_tensor = recon - T
            
            # Gradients
            grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', err_tensor, V_r, W_r)
            grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', err_tensor, U_r, W_r)
            grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', err_tensor, U_r, V_r)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        # Run optimization
        # Use L-BFGS-B. 
        # Increase maxiter and maxfun slightly if time permits, but keep it bounded.
        # The bottleneck is usually the number of function evaluations.
        # 100-200 iterations is standard.
        opt_res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 150, 'maxfun': 300, 'disp': False})
        
        if time.time() - start_time > time_limit:
            break
            
        # Extract optimized real values
        x_opt = opt_res.x
        U_opt = x_opt[:U_flat.size].reshape(n * m, R)
        V_opt = x_opt[U_flat.size:U_flat.size + V_flat.size].reshape(m * p, R)
        W_opt = x_opt[U_flat.size + V_flat.size:].reshape(p * n, R)
        
        # 2. Round to nearest half-integer
        # Multiply by 2, round, divide by 2
        U_int = np.round(U_opt * 2) / 2.0
        V_int = np.round(V_opt * 2) / 2.0
        W_int = np.round(W_opt * 2) / 2.0
        
        # 3. Verify and Local Search
        err = _compute_decomposition_error(U_int, V_int, W_int, T)
        
        # If not exact, try local search from this point
        if err > 1e-10:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 1.0:
                err = _local_search(U_int, V_int, W_int, T, values, max_iter=10000, timeout=remaining_time)
                
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U_int.copy(), V_int.copy(), W_int.copy()
            
        if best_error < 1e-10:
            break
            
    if best_error < 1e-10:
        return best_U, best_V, best_W
    return None

from scipy.optimize import minimize

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
    # Expand range slightly to catch larger values if needed, but 2.0 is usually enough for small matrices
    values = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Try ranks lower than schoolbook
    # Known upper bound for 2x3x4 is likely around 18-20.
    # Let's try from 14 up to 23.
    # Starting lower might find a better rank if it exists.
    start_time = time.time()
    deadline = 145.0 # Safety margin before 150s kill
    
    # We want to find the MINIMAL rank.
    # So we should iterate from low to high.
    # If we find a valid decomposition at rank R, we should check if R-1 is possible?
    # The loop goes 14, 15, 16...
    # If we find 16, we have already checked 14, 15. So 16 is the min in this range.
    # We can break.
    
    for R in range(14, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        # Time budget for this rank
        # We want to give more time to lower ranks? Or equal time?
        # Equal time is safer.
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        # Cap time per rank to avoid spending too long on one if many fail
        # But if we are near the solution, we might need more time.
        # Let's cap at 15s per rank.
        time_budget = min(time_budget, 15.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify again to be safe
            err = _compute_decomposition_error(U, V, W, T)
            if err < 1e-10:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                # Found a valid decomposition. 
                # Since we are iterating from low to high, this is the minimal one we found.
                # We can break.
                break
                
    # If no better found, return schoolbook
    return best_result
