import numpy as np
from functools import lru_cache

# Global cache for pre-computed policies
_policy_cache = {}

def _compute_policy(params):
    """
    Computes the optimal bid and ask depths for all steps and inventory levels.
    Returns a dictionary with 'bid' and 'ask' arrays of shape (steps, 2*Q+1).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    alpha = A * dt
    
    # Grid of depths to consider
    # We want to cover a reasonable range. 
    # If k is small, depths can be larger.
    # Let's use a log-spaced or linear grid. Linear is simpler.
    # Range: 0.01 to 5.0
    d_grid = np.linspace(0.01, 5.0, 200)
    
    # Pre-compute P(d) and P(d)*d for the grid
    # P(d) = 1 - exp(-alpha * exp(-k * d))
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-alpha * exp_term)
    Pd_grid = P_grid * d_grid
    
    # Value function V(n, q)
    # V_next is an array of size 2*Q+1
    # Initialize V_next to 0 for n = steps
    V_next = np.zeros(2 * Q + 1)
    
    bid_depths = np.zeros((steps, 2 * Q + 1))
    ask_depths = np.zeros((steps, 2 * Q + 1))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        
        # Inventory cost
        # q values: -Q, ..., Q
        q_vals = np.arange(-Q, Q + 1)
        inv_cost = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt
        
        # For each q, we need to find optimal d_b and d_a
        # We can vectorize over q
        
        # Bid optimization:
        # Maximize P(d) * (d + V_next[q+1])
        # V_next indices: q is index i in 0..2Q. q_val = i - Q.
        # Next inventory for bid fill (buying) is q+1.
        # Index for q+1 is i+1.
        # Valid if q < Q. If q == Q, no bid fill allowed.
        
        # Ask optimization:
        # Maximize P(d) * (d + V_next[q-1])
        # Next inventory for ask fill (selling) is q-1.
        # Index for q-1 is i-1.
        # Valid if q > -Q. If q == -Q, no ask fill allowed.
        
        # We process all q at once.
        # V_next_plus = V_next shifted left (for q+1)
        # V_next_minus = V_next shifted right (for q-1)
        
        V_next_plus = np.zeros(2 * Q + 1)
        V_next_minus = np.zeros(2 * Q + 1)
        
        # For q = -Q .. Q-1, next q is q+1. Index i -> i+1
        V_next_plus[:-1] = V_next[1:]
        # For q = Q, no bid fill. We can set V_next_plus[-1] to something, but we will mask it.
        
        # For q = -Q+1 .. Q, next q is q-1. Index i -> i-1
        V_next_minus[1:] = V_next[:-1]
        # For q = -Q, no ask fill.
        
        # Calculate objective for all d in grid for all q
        # Shape: (200, 2Q+1)
        # P_grid[:, None] * (d_grid[:, None] + V_next_plus[None, :])
        
        obj_bid = P_grid[:, None] * (d_grid[:, None] + V_next_plus[None, :])
        obj_ask = P_grid[:, None] * (d_grid[:, None] + V_next_minus[None, :])
        
        # Find max index
        max_idx_bid = np.argmax(obj_bid, axis=0)
        max_idx_ask = np.argmax(obj_ask, axis=0)
        
        # Get optimal depths
        optimal_d_bid = d_grid[max_idx_bid]
        optimal_d_ask = d_grid[max_idx_ask]
        
        # Get max values
        max_val_bid = obj_bid[max_idx_bid, np.arange(2 * Q + 1)]
        max_val_ask = obj_ask[max_idx_ask, np.arange(2 * Q + 1)]
        
        # Apply constraints:
        # If q == Q, no bid fill. Set depth to inf (or large number) and value to 0?
        # The problem says "no bid fill at q=Q". This means we cannot quote bid.
        # So the term for bid fill is 0.
        # Similarly for ask at q=-Q.
        
        # Mask for bid: q < Q. Index 0 to 2Q-1. Last index 2Q is Q.
        bid_mask = np.arange(2 * Q + 1) < 2 * Q
        ask_mask = np.arange(2 * Q + 1) > 0
        
        # If masked, the value contribution is 0, and depth is irrelevant (set to 0 or inf).
        # The problem says "np.inf means no quote".
        
        # Update V_curr
        # V_curr[i] = -inv_cost[i] + (bid_val if bid_mask[i] else 0) + (ask_val if ask_mask[i] else 0)
        
        V_curr = -inv_cost
        V_curr[bid_mask] += max_val_bid[bid_mask]
        V_curr[ask_mask] += max_val_ask[ask_mask]
        
        # Store depths
        bid_depths[n, :] = optimal_d_bid
        ask_depths[n, :] = optimal_d_ask
        
        # Set no-quote depths to inf
        bid_depths[n, ~bid_mask] = np.inf
        ask_depths[n, ~ask_mask] = np.inf
        
        V_next = V_curr
        
    return {'bid': bid_depths, 'ask': ask_depths}

def _get_params_key(params):
    return (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])

def quote(t, inventory, mid, params):
    key = _get_params_key(params)
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    
    policy = _policy_cache[key]
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step index
    # t = n * dt. dt = T / steps.
    # n = t / dt = t * steps / T
    n = int(np.round(t * params['steps'] / params['T']))
    # Clamp n to valid range [0, steps-1]
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to indices
    # inventory is in [-Q, Q]
    # Index = inventory + Q
    indices = inventory + Q
    indices = np.clip(indices, 0, 2 * Q)
    
    bid_depths = policy['bid'][n, indices]
    ask_depths = policy['ask'][n, indices]
    
    return bid_depths, ask_depths
