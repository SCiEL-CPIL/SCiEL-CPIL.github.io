import numpy as np
import time

_DEADLINE = time.time() + 290
_POLICY_CACHE = {}

def _solve_dp(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Depth grid
    # Optimal depth is usually small. 10/k covers a wide range.
    # We use 200 points for good resolution.
    max_depth = 10.0 / k
    grid_size = 200
    depths = np.linspace(0.0, max_depth, grid_size)
    
    # Precompute probabilities for the grid
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # Note: exp(-k*d) is monotonic decreasing.
    exp_term = np.exp(-k * depths)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # V_next[q_index] where q_index = q + Q
    num_states = 2 * Q + 1
    V_next = np.zeros(num_states)
    
    bid_table = np.full((steps, num_states), np.inf)
    ask_table = np.full((steps, num_states), np.inf)
    
    # Inventory indices: 0 corresponds to q=-Q, ..., Q corresponds to q=Q
    # q_values = np.arange(-Q, Q+1)
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_states)
        
        # Vectorized calculation for all q
        # We need to compute max over d for each q.
        # Since the function to maximize depends on V_next[q+1], V_next[q], V_next[q-1],
        # we can do this in parallel for all q.
        
        # Create arrays for the terms
        # V_next is shape (num_states,)
        # For bid:
        # If q < Q: Maximize p(d)*(d + V_next[q+1]) + (1-p(d))*V_next[q]
        # If q == Q: Value is V_next[q], depth is inf
        
        # Let's handle the interior states first (indices 0 to num_states-2 for bid, 1 to num_states-1 for ask)
        
        # --- Bid Optimization ---
        # Indices 0 to num_states-2 correspond to q = -Q to Q-1
        # q+1 index is idx + 1
        # q index is idx
        
        idx_bid = np.arange(num_states - 1) # 0 to 2Q-1
        V_bid_next = V_next[idx_bid + 1] # V_next[q+1]
        V_bid_stay = V_next[idx_bid]     # V_next[q]
        
        # Objective: p_grid * (depths + V_bid_next) + (1 - p_grid) * V_bid_stay
        # Shape: (grid_size, num_states-1)
        # We want max over axis 0 (depths)
        
        # To avoid huge memory if grid_size is large, we can do it in chunks or just use broadcasting.
        # 200 * 60 is small.
        
        term_bid = p_grid[:, None] * (depths[:, None] + V_bid_next[None, :]) + (1.0 - p_grid[:, None]) * V_bid_stay[None, :]
        max_val_bid = np.max(term_bid, axis=0)
        best_idx_bid = np.argmax(term_bid, axis=0)
        best_depth_bid = depths[best_idx_bid]
        
        # Assign to table
        # bid_table[n, 0:num_states-1] = best_depth_bid
        # For q=Q (index num_states-1), depth is inf, value is V_next[num_states-1]
        
        bid_table[n, :num_states-1] = best_depth_bid
        bid_table[n, num_states-1] = np.inf
        
        # Value contribution from bid
        val_bid = np.zeros(num_states)
        val_bid[:num_states-1] = max_val_bid
        val_bid[num_states-1] = V_next[num_states-1]
        
        # --- Ask Optimization ---
        # Indices 1 to num_states-1 correspond to q = -Q+1 to Q
        # q-1 index is idx - 1
        # q index is idx
        
        idx_ask = np.arange(1, num_states) # 1 to 2Q
        V_ask_next = V_next[idx_ask - 1] # V_next[q-1]
        V_ask_stay = V_next[idx_ask]     # V_next[q]
        
        term_ask = p_grid[:, None] * (depths[:, None] + V_ask_next[None, :]) + (1.0 - p_grid[:, None]) * V_ask_stay[None, :]
        max_val_ask = np.max(term_ask, axis=0)
        best_idx_ask = np.argmax(term_ask, axis=0)
        best_depth_ask = depths[best_idx_ask]
        
        # Assign to table
        # ask_table[n, 1:num_states] = best_depth_ask
        # For q=-Q (index 0), depth is inf, value is V_next[0]
        
        ask_table[n, 1:num_states] = best_depth_ask
        ask_table[n, 0] = np.inf
        
        # Value contribution from ask
        val_ask = np.zeros(num_states)
        val_ask[1:num_states] = max_val_ask
        val_ask[0] = V_next[0]
        
        # --- Total Value ---
        # Penalty: 0.5 * gamma * sigma^2 * q^2 * dt
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        penalty = 0.5 * gamma * (sigma ** 2) * (q_vals ** 2) * dt
        
        V_curr = val_bid + val_ask - penalty
        
        V_next = V_curr
        
    return bid_table, ask_table

def quote(t, inventory, mid, params):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple heuristic if we run out of time (unlikely with vectorized DP)
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        steps = params['steps']
        T = params['T']
        dt = T / steps
        remaining_time = T - t
        time_fraction = remaining_time / T
        q = inventory.astype(np.float64)
        base_depth = 1.0 / k
        beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
        q_scaled = q / Q
        q_adjust = np.tanh(1.5 * q_scaled)
        bid_depth = base_depth + beta * q_adjust
        ask_depth = base_depth - beta * q_adjust
        bid_depth = np.maximum(bid_depth, 0.01)
        ask_depth = np.maximum(ask_depth, 0.01)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth

    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _POLICY_CACHE:
        # Check if we have time to compute
        if time.time() < _DEADLINE - 10: # 10s margin
            _POLICY_CACHE[key] = _solve_dp(params)
        else:
            # Fallback
            sigma = params['sigma']
            A = params['A']
            k = params['k']
            gamma = params['gamma']
            Q = params['Q']
            steps = params['steps']
            T = params['T']
            dt = T / steps
            remaining_time = T - t
            time_fraction = remaining_time / T
            q = inventory.astype(np.float64)
            base_depth = 1.0 / k
            beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
            q_scaled = q / Q
            q_adjust = np.tanh(1.5 * q_scaled)
            bid_depth = base_depth + beta * q_adjust
            ask_depth = base_depth - beta * q_adjust
            bid_depth = np.maximum(bid_depth, 0.01)
            ask_depth = np.maximum(ask_depth, 0.01)
            bid_depth[inventory >= Q] = np.inf
            ask_depth[inventory <= -Q] = np.inf
            return bid_depth, ask_depth
            
    bid_table, ask_table = _POLICY_CACHE[key]
    
    # Determine current step index
    steps = params['steps']
    T = params['T']
    dt = T / steps
    n = int(t / dt)
    # Clamp n to [0, steps-1]
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    Q = params['Q']
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    indices = inventory + Q
    
    # Get depths
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    return bid_depths, ask_depths
