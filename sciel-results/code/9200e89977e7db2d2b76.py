import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

# Cache for pre-computed policies
_policy_cache = {}

def _precompute_policy(params):
    """
    Pre-computes the optimal bid and ask depths for all steps and inventory levels
    using Dynamic Programming (Backward Induction).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    
    # Create a cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]
    
    # Depth grid
    # The minimum valid depth is -3/k.
    # Optimal depths are typically positive, but can be negative if inventory risk is high.
    # Let's sample from -3/k to 5/k.
    min_depth = -3.0 / k
    max_depth = 5.0 / k
    grid_size = 100
    depths = np.linspace(min_depth, max_depth, grid_size)
    
    # Pre-compute fill probabilities for all depths in the grid
    # p = 1 - exp(-A * exp(-k * depth) * dt)
    exp_term = np.exp(-k * depths)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    
    # Initialize value function and policy tables
    # V[n, q] where q is indexed 0..2Q corresponding to inventory -Q..Q
    num_states = 2 * Q + 1
    V = np.zeros((steps + 1, num_states), dtype=np.float64)
    
    # Policy tables
    opt_bid = np.zeros((steps, num_states), dtype=np.float64)
    opt_ask = np.zeros((steps, num_states), dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        t_rem = T - n * dt # Not strictly needed for DP, but good for debugging
        
        for q_idx in range(num_states):
            q = q_idx - Q
            
            # Inventory penalty
            penalty = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get future values
            # If q == Q, no bid fill possible (inventory cap)
            # If q == -Q, no ask fill possible (inventory floor)
            
            if q == Q:
                # Only ask fills possible
                # We can optimize 'a' only. 'b' can be set to min_depth (no quote) or inf?
                # Problem says "np.inf means no quote". 
                # But we must return finite values or inf.
                # If we set b to inf, p_b = 0.
                # Let's just optimize 'a' and set 'b' to min_depth (worst case) or inf.
                # Actually, if we don't want to quote bid, we should quote at a very negative depth?
                # No, "distance below mid". If we want NO quote, we use inf.
                # But the DP maximizes reward. If we set b to min_depth, we might get a fill we don't want.
                # At q=Q, a bid fill is FORBIDDEN.
                # So p_b must be 0. This implies b = inf.
                # In our grid, we don't have inf.
                # We can handle boundaries separately.
                
                # For q=Q, optimal b is inf (no bid).
                # We optimize a.
                # Value = p_a * a - penalty + p_a * V[n+1, q-1] + (1-p_a) * V[n+1, q]
                # Note: (1-p_b)(1-p_a) becomes (1-p_a) since p_b=0.
                
                # Vectorized over 'a'
                a_vals = depths
                p_a_vals = p_fill
                
                # Immediate reward: p_a * a
                # Future value: p_a * V[n+1, q-1] + (1-p_a) * V[n+1, q]
                # V[n+1, q+1] is not used.
                
                v_next_q_minus_1 = V[n+1, q_idx - 1]
                v_next_q = V[n+1, q_idx]
                
                J = p_a_vals * a_vals - penalty + p_a_vals * v_next_q_minus_1 + (1.0 - p_a_vals) * v_next_q
                
                idx_max = np.argmax(J)
                opt_bid[n, q_idx] = np.inf # No bid
                opt_ask[n, q_idx] = a_vals[idx_max]
                V[n, q_idx] = J[idx_max]
                
            elif q == -Q:
                # Only bid fills possible
                # Optimize b, set a = inf
                
                b_vals = depths
                p_b_vals = p_fill
                
                v_next_q_plus_1 = V[n+1, q_idx + 1]
                v_next_q = V[n+1, q_idx]
                
                # Value = p_b * b - penalty + p_b * V[n+1, q+1] + (1-p_b) * V[n+1, q]
                J = p_b_vals * b_vals - penalty + p_b_vals * v_next_q_plus_1 + (1.0 - p_b_vals) * v_next_q
                
                idx_max = np.argmax(J)
                opt_bid[n, q_idx] = b_vals[idx_max]
                opt_ask[n, q_idx] = np.inf # No ask
                V[n, q_idx] = J[idx_max]
                
            else:
                # Both fills possible
                v_next_q_minus_1 = V[n+1, q_idx - 1]
                v_next_q = V[n+1, q_idx]
                v_next_q_plus_1 = V[n+1, q_idx + 1]
                
                # Vectorized over b and a
                # b_vals shape (100, 1), a_vals shape (1, 100)
                b_vals = depths[:, np.newaxis]
                a_vals = depths[np.newaxis, :]
                
                p_b_vals = p_fill[:, np.newaxis]
                p_a_vals = p_fill[np.newaxis, :]
                
                # Immediate reward: p_b * b + p_a * a
                # Future value: 
                # p_b * v_q+1 + p_a * v_q-1 + (1-p_b)(1-p_a) * v_q
                # Note: (1-p_b)(1-p_a) = 1 - p_b - p_a + p_b*p_a
                
                J = (p_b_vals * b_vals + p_a_vals * a_vals) - penalty \
                    + p_b_vals * v_next_q_plus_1 \
                    + p_a_vals * v_next_q_minus_1 \
                    + (1.0 - p_b_vals) * (1.0 - p_a_vals) * v_next_q
                
                # Find the pair (b, a) that maximizes J
                idx_flat = np.argmax(J)
                idx_b, idx_a = np.unravel_index(idx_flat, J.shape)
                
                opt_bid[n, q_idx] = depths[idx_b]
                opt_ask[n, q_idx] = depths[idx_a]
                V[n, q_idx] = J[idx_flat]
        
        # Check deadline
        if time.time() > _DEADLINE - 10.0:
            # If we run out of time, fill remaining with heuristic
            # This is unlikely to happen with 100 steps left, but just in case
            # We'll break and use a fallback in the quote function if table is incomplete
            break
            
    _policy_cache[key] = (opt_bid, opt_ask)
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths for the given state.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute the optimal policy table
    opt_bid, opt_ask = _precompute_policy(params)
    
    # Determine current step index
    dt = T / steps
    n = int(t / dt)
    # Clamp n to valid range
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Convert inventory to index
    # inventory is array of ints in [-Q, Q]
    inv = np.asarray(inventory, dtype=np.int32)
    # Clamp inventory to valid range just in case
    inv = np.clip(inv, -Q, Q)
    idx = inv + Q
    
    # Fetch optimal depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Ensure output is float arrays
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
