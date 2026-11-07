"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies per regime
_policy_cache = {}

def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via vectorized dynamic programming.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)

    # Value function
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Grid of depths
    # Include a very large depth to approximate 'no quote' (p ~ 0)
    d_min = 0.01
    d_max = 20.0 / k  # exp(-20) is ~ 2e-9, so p is very small
    n_depths = 100
    depth_grid = np.linspace(d_min, d_max, n_depths)
    
    # Precompute probabilities
    # p_fill = 1 - exp(-A * exp(-k * d) * dt)
    lam = A * np.exp(-k * depth_grid) * dt
    p_fill = 1.0 - np.exp(-lam)
    
    # Indices for "no quote" (max depth)
    idx_no_quote = n_depths - 1
    
    # Precompute depth values for broadcasting
    # d_b shape: (n_depths, 1, 1)
    # d_a shape: (1, n_depths, 1)
    d_b_grid = depth_grid[:, np.newaxis, np.newaxis]
    d_a_grid = depth_grid[np.newaxis, :, np.newaxis]
    
    # p_b shape: (n_depths, 1, 1)
    # p_a shape: (1, n_depths, 1)
    p_b_grid = p_fill[:, np.newaxis, np.newaxis]
    p_a_grid = p_fill[np.newaxis, :, np.newaxis]
    
    # State-dependent arrays will be broadcast to (1, 1, n_states)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Calculate state-dependent terms
        # Delta_b[i] = V_next[i+1] - V_next[i]
        # Delta_a[i] = V_next[i-1] - V_next[i]
        # Curv[i] = V_next[i+1] + V_next[i-1] - 2*V_next[i]
        
        # Handle boundaries for Delta and Curv
        # For i=0, i-1 is invalid. But we will mask out invalid actions later.
        # For calculation purposes, we can pad V_next or use np.roll/clip?
        # Let's just create padded V to avoid index errors in vectorized ops
        V_pad = np.zeros(n_states + 2)
        V_pad[1:-1] = V_next
        # Boundary values: We don't really care if they are wrong because we mask actions.
        # But to be safe, let's set V_pad[0] = V_next[0] and V_pad[-1] = V_next[-1]
        V_pad[0] = V_next[0]
        V_pad[-1] = V_next[-1]
        
        V_i = V_next[np.newaxis, np.newaxis, :] # (1, 1, n_states)
        
        # V_i_plus = V_next[i+1]
        V_i_plus = V_pad[2:][np.newaxis, np.newaxis, :] # (1, 1, n_states)
        # V_i_minus = V_next[i-1]
        V_i_minus = V_pad[:-2][np.newaxis, np.newaxis, :] # (1, 1, n_states)
        
        # Check shapes: V_i_plus for i=0 is V_next[1]. Correct.
        # V_i_minus for i=0 is V_next[-1]? No, V_pad[:-2] for i=0 is V_pad[0] which is V_next[0].
        # Wait.
        # i=0: V_i_minus should be V_next[-1] (invalid).
        # My padding: V_pad = [V_next[0], V_next[0], V_next[1], ..., V_next[Q], V_next[-1]]?
        # Let's be precise.
        # V_next indices: 0..N-1
        # V_i_plus[i] = V_next[i+1]. For i=N-1, this is out of bounds.
        # V_i_minus[i] = V_next[i-1]. For i=0, this is out of bounds.
        
        # Let's construct V_i_plus and V_i_minus carefully.
        V_i_plus = np.zeros(n_states)
        V_i_minus = np.zeros(n_states)
        
        # i=0 to N-2: V_next[i+1]
        V_i_plus[0:N-1] = V_next[1:N]
        # i=N-1: No next state (boundary). Value is irrelevant if masked.
        V_i_plus[N-1] = V_next[N-1]
        
        # i=1 to N-1: V_next[i-1]
        V_i_minus[1:N] = V_next[0:N-1]
        # i=0: No prev state (boundary). Value is irrelevant if masked.
        V_i_minus[0] = V_next[0]
        
        Delta_b = V_i_plus - V_next
        Delta_a = V_i_minus - V_next
        Curv = V_i_plus + V_i_minus - 2 * V_next
        
        Cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt
        
        # Broadcast to (1, 1, n_states)
        Delta_b_b = Delta_b[np.newaxis, np.newaxis, :]
        Delta_a_b = Delta_a[np.newaxis, np.newaxis, :]
        Curv_b = Curv[np.newaxis, np.newaxis, :]
        Cost_b = Cost[np.newaxis, np.newaxis, :]
        V_i_b = V_next[np.newaxis, np.newaxis, :]
        
        # Calculate Value Tensor (n_depths, n_depths, n_states)
        # V = p_b*(d_b + Delta_b) + p_a*(d_a + Delta_a) + V_i - Cost - p_b*p_a*Curv
        
        # Term 1: p_b * (d_b + Delta_b)
        term_b = p_b_grid * (d_b_grid + Delta_b_b)
        
        # Term 2: p_a * (d_a + Delta_a)
        term_a = p_a_grid * (d_a_grid + Delta_a_b)
        
        # Term 3: V_i - Cost
        term_base = V_i_b - Cost_b
        
        # Term 4: - p_b * p_a * Curv
        term_coupling = - (p_b_grid * p_a_grid) * Curv_b
        
        V_full = term_b + term_a + term_base + term_coupling
        
        # Apply Boundary Masks
        # For i=0 (q=-Q), no ask fill. Force d_a to be the "no quote" depth (idx_no_quote).
        # Set V_full[:, :, 0] to -inf for all d_a except idx_no_quote.
        # Actually, we want to pick the best d_b given p_a ~ 0.
        # So we keep the column d_a = idx_no_quote and mask others.
        
        # Mask for i=0
        mask_a_invalid = np.ones((n_depths, n_states), dtype=bool)
        mask_a_invalid[idx_no_quote, :] = False # Keep this one
        # We need to apply this mask to the (d_a, i) dimensions?
        # V_full shape: (d_b, d_a, i)
        # We want to mask d_a for i=0.
        # Create a mask of shape (1, n_depths, n_states)
        mask_a = np.ones((1, n_depths, n_states), dtype=bool)
        # For i=0, mask all d_a except idx_no_quote
        mask_a[0, idx_no_quote, 0] = False # This is False? No, we want to KEEP it.
        # Let's use -inf logic.
        
        # Easier:
        V_full[:, :, 0] = -np.inf
        V_full[:, idx_no_quote, 0] = V_full[:, idx_no_quote, 0] # Restore the valid column
        
        # For i=N-1 (q=Q), no bid fill. Force d_b to be idx_no_quote.
        V_full[:, :, N-1] = -np.inf
        V_full[idx_no_quote, :, N-1] = V_full[idx_no_quote, :, N-1] # Restore valid row
        
        # Find best actions
        # Reshape to (n_depths*n_depths, n_states)
        V_flat = V_full.reshape(n_depths * n_depths, n_states)
        
        # Max over depth axis (axis 0) for each state
        best_flat_idx = np.argmax(V_flat, axis=0) # Shape (n_states,)
        
        # Best values
        V_curr = V_flat[best_flat_idx, np.arange(n_states)]
        
        # Map back to d_b, d_a indices
        best_b_idx = best_flat_idx // n_depths
        best_a_idx = best_flat_idx % n_depths
        
        # Store actions
        opt_bid[n, :] = depth_grid[best_b_idx]
        opt_ask[n, :] = depth_grid[best_a_idx]
        
        # Update V
        V_next = V_curr

    return opt_bid, opt_ask

def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    steps = params['steps']
    Q = params['Q']
    
    # Get or compute policy
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        # Fallback to seed behavior
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()
    
    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    idx = inventory + Q
    idx = np.clip(idx, 0, 2*Q)
    
    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]
    
    # Ensure valid outputs
    # If depth is inf, it means no quote.
    # If depth is 0, it means quote at mid.
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
