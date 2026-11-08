import time
import numpy as np
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s for validation and return
    
    # Try rank 64
    try:
        result_64 = _try_rank_64(n, m, p, deadline)
        if result_64 is not None:
            return result_64
    except Exception:
        pass
        
    # Fallback to rank 66
    return _construct_66(n, m, p)

def _try_rank_64(n, m, p, deadline):
    # Search for rank 64 solution using gradient descent
    # This is a heuristic search.
    random.seed(42)
    
    rank = 64
    
    # Precompute index arrays for vectorized operations
    # We need to compute: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and gradients.
    
    # Let's flatten indices
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # Create index arrays for all (i, j, k) combinations
    i_vals = np.arange(n)
    j_vals = np.arange(m)
    k_vals = np.arange(p)
    
    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()
    kk_flat = kk.flatten()
    
    # Compute linear indices
    u_idx_flat = ii_flat * m + jj_flat  # shape (n*m*p,)
    v_idx_flat = jj_flat * p + kk_flat  # shape (n*m*p,)
    w_idx_flat = kk_flat * n + ii_flat  # shape (n*m*p,)
    
    # Target tensor is all 1s
    target = np.ones(n * m * p)
    
    best_solution = None
    best_loss = float('inf')
    
    # Try a few random restarts
    max_restarts = 5
    time_per_restart = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize U, V, W
        # Use small random initialization
        scale = 0.1
        U = np.random.randn(n*m, rank) * scale
        V = np.random.randn(m*p, rank) * scale
        W = np.random.randn(p*n, rank) * scale
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        steps = 100
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
                
            # Compute T_flat = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            # This is: for each t in range(n*m*p), T[t] = sum_r U[u_idx[t], r] * V[v_idx[t], r] * W[w_idx[t], r]
            
            # We can vectorize this by looping over r
            # T_flat = zeros(n*m*p)
            # for r in range(rank):
            #     T_flat += U[u_idx_flat, r] * V[v_idx_flat, r] * W[w_idx_flat, r]
            
            # This loop over rank (64) with vectorized ops on arrays of size n*m*p (72) is fast.
            T_flat = np.zeros(n * m * p)
            for r in range(rank):
                T_flat += U[u_idx_flat, r] * V[v_idx_flat, r] * W[w_idx_flat, r]
                
            residual = T_flat - target
            loss = np.sum(residual ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
                
            if loss < 1e-10:
                # Found exact solution
                return _format_solution(n, m, p, U, V, W)
                
            # Compute gradients
            # grad_U[u, r] = sum_{t: u_idx[t]=u} residual[t] * V[v_idx[t], r] * W[w_idx[t], r]
            # grad_V[v, r] = sum_{t: v_idx[t]=v} residual[t] * U[u_idx[t], r] * W[w_idx[t], r]
            # grad_W[w, r] = sum_{t: w_idx[t]=w} residual[t] * U[u_idx[t], r] * V[v_idx[t], r]
            
            # We can compute these by scattering
            # For efficiency, we can use np.add.at or similar, but looping over r is okay.
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Precompute products for each r
            # For each r:
            #   term_U_contrib[t] = residual[t] * V[v_idx[t], r] * W[w_idx[t], r]
            #   term_V_contrib[t] = residual[t] * U[u_idx[t], r] * W[w_idx[t], r]
            #   term_W_contrib[t] = residual[t] * U[u_idx[t], r] * V[v_idx[t], r]
            
            for r in range(rank):
                # Compute contributions
                # Note: U[u_idx_flat, r] is a vector of length n*m*p
                u_vals = U[u_idx_flat, r]
                v_vals = V[v_idx_flat, r]
                w_vals = W[w_idx_flat, r]
                
                # grad_U contribution: scatter residual * v_vals * w_vals to U indices
                contrib_U = residual * v_vals * w_vals
                # Scatter to grad_U
                # grad_U[u_idx_flat, r] += contrib_U
                # Using np.add.at
                np.add.at(grad_U, (u_idx_flat, r), contrib_U)
                
                # grad_V contribution: scatter residual * u_vals * w_vals to V indices
                contrib_V = residual * u_vals * w_vals
                np.add.at(grad_V, (v_idx_flat, r), contrib_V)
                
                # grad_W contribution: scatter residual * u_vals * v_vals to W indices
                contrib_W = residual * u_vals * v_vals
                np.add.at(grad_W, (w_idx_flat, r), contrib_W)
                
            # Update with momentum
            vU = momentum * vU - lr * grad_U
            vV = momentum * vV - lr * grad_V
            vW = momentum * vW - lr * grad_W
            
            U += vU
            V += vV
            W += vW
            
            # Optional: decay learning rate
            # lr = 0.99 * lr
            
    if best_solution is not None and best_loss < 1e-8:
        U, V, W = best_solution
        return _format_solution(n, m, p, U, V, W)
        
    return None

def _format_solution(n, m, p, U, V, W):
    # Convert to list of lists as required by interface?
    # The interface says "factor matrices of shapes... with integer/half-integer entries"
    # The current program returns lists of lists.
    # Let's check if numpy arrays are accepted. The task says "integer/half-integer entries".
    # If the evaluator expects exact integers/half-integers, random float results won't work.
    # The fallback _construct_66 returns lists of lists with 1, -1, 0.
    # So _try_rank_64 must also return exact values if it succeeds.
    # Since gradient descent produces floats, we would need to snap to nearest half-integer?
    # Or maybe the evaluator accepts floats that are close to integers?
    # "exactly decompose" suggests exactness is required.
    # So if we find a solution via GD, it must be extremely close to integer/half-integer.
    # Let's round to nearest 0.5 and check.
    
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Check if rounded solution is still valid
    # Recompute tensor
    u_idx_flat, v_idx_flat, w_idx_flat = _get_indices(n, m, p)
    T_flat = np.zeros(n * m * p)
    rank = U_rounded.shape[1]
    for r in range(rank):
        T_flat += U_rounded[u_idx_flat, r] * V_rounded[v_idx_flat, r] * W_rounded[w_idx_flat, r]
        
    if np.allclose(T_flat, np.ones(n*m*p), atol=1e-9):
        # Convert to lists
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return None

def _get_indices(n, m, p):
    i_vals = np.arange(n)
    j_vals = np.arange(m)
    k_vals = np.arange(p)
    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()
    kk_flat = kk.flatten()
    u_idx_flat = ii_flat * m + jj_flat
    v_idx_flat = jj_flat * p + kk_flat
    w_idx_flat = kk_flat * n + ii_flat
    return u_idx_flat, v_idx_flat, w_idx_flat

def _construct_66(n, m, p):
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
        
    return {"U": U, "V": V, "W": W}
