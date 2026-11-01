import numpy as np
import time

def construct():
    """
    Decompose 2x2 * 2x10 multiplication.
    Start with Strassen rank-35 baseline, then attempt to find lower rank
    via numerical search with deadline guard.
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = 2, 2, 10
    
    # First, build the rank-35 Strassen baseline
    U_base, V_base, W_base = _strassen_baseline(n, m, p)
    
    # Try to find lower rank decompositions
    best_rank = 35
    best_U, best_V, best_W = U_base, V_base, W_base
    
    # Try ranks from 20 upward, stop if time is running low
    for rank in range(20, best_rank):
        if time.time() - start_time > deadline - 30:
            break
        
        result = _try_find_decomposition(n, m, p, rank, start_time, deadline)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = rank
            best_U, best_V, best_W = U_found, V_found, W_found
            break  # Found a valid decomposition at this rank, stop
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _strassen_baseline(n, m, p):
    """Build the rank-35 Strassen decomposition."""
    num_blocks = p // 2
    block_rank = 7
    total_rank = num_blocks * block_rank
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    u_terms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    v_terms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_terms = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1}
    ]
    
    for block_idx in range(num_blocks):
        base_col = block_idx * block_rank
        b_col_offset = 2 * block_idx
        c_col_offset = 2 * block_idx
        
        for t in range(block_rank):
            col = base_col + t
            U[0:4, col] = u_terms[t]
            
            v_vals = v_terms[t]
            V[0 * p + (b_col_offset + 0), col] = v_vals[0]
            V[0 * p + (b_col_offset + 1), col] = v_vals[1]
            V[1 * p + (b_col_offset + 0), col] = v_vals[2]
            V[1 * p + (b_col_offset + 1), col] = v_vals[3]
            
            for local_c_idx, val in w_terms[t].items():
                if local_c_idx == 0:
                    i = 0; k_local = 0
                elif local_c_idx == 1:
                    i = 1; k_local = 0
                elif local_c_idx == 2:
                    i = 0; k_local = 1
                else:
                    i = 1; k_local = 1
                
                global_k = c_col_offset + k_local
                global_w_idx = global_k * n + i
                W[global_w_idx, col] = val
    
    return U, V, W

def _try_find_decomposition(n, m, p, rank, start_time, deadline):
    """
    Try to find a rank-r decomposition using numerical optimization.
    Returns (U, V, W) if found, None otherwise.
    """
    # Time budget for this rank attempt
    time_budget = min(20.0, deadline - (time.time() - start_time) - 10.0)
    if time_budget <= 5.0:
        return None
    
    # Initialize with random matrices
    np.random.seed(rank)  # Reproducible
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # The constraint is: for all i,j,k: sum_r U[i,r] * V[j,r] * W[k,r] = delta_{i,j,k}
    # where delta is 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise.
    
    # Build the target tensor
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor M[i,j,k] = 1 if this is a valid term
                target[i, j, k] = 1.0
    
    # Use scipy.optimize to minimize the error
    from scipy.optimize import minimize
    
    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_params(params):
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:]
        U_r = U_flat.reshape(n * m, rank)
        V_r = V_flat.reshape(m * p, rank)
        W_r = W_flat.reshape(p * n, rank)
        return U_r, V_r, W_r
    
    def objective(params):
        U_r, V_r, W_r = unflatten_params(params)
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Efficient computation: for each r, outer product
        T = np.zeros((n, m, p))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U_r[:, r], V_r[:, r], W_r[:, r])
        error = np.sum((T - target) ** 2)
        return error
    
    # Initial objective
    init_params = flatten_params(U, V, W)
    init_obj = objective(init_params)
    
    # If initial is already good enough, we're done
    if init_obj < 1e-10:
        U_r, V_r, W_r = unflatten_params(init_params)
        return U_r, V_r, W_r
    
    # Use L-BFGS-B with time limit
    try:
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000, 'disp': False}
        )
        
        if result.success or result.fun < 1e-8:
            U_r, V_r, W_r = unflatten_params(result.x)
            # Verify
            T = np.zeros((n, m, p))
            for r in range(rank):
                T += np.einsum('i,j,k->ijk', U_r[:, r], V_r[:, r], W_r[:, r])
            if np.max(np.abs(T - target)) < 1e-6:
                return U_r, V_r, W_r
    except Exception:
        pass
    
    return None
