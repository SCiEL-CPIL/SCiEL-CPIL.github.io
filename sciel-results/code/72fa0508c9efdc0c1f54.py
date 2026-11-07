"""
Strategy: Stationary Policy Approximation with Inventory Correction.
Instead of solving the full 1000-step finite-horizon DP (which limits depth grid resolution),
we solve for the stationary optimal policy (time-invariant depths) using Value Iteration.
This allows for a much finer depth grid (e.g., 1000+ points) within the time budget,
reducing discretization error in the depth selection.
We add a heuristic time-dependent shift to the inventory state to approximate
the finite-horizon effect (drifting back towards zero inventory as t -> T).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin

def _compute_stationary_policy(params):
    """
    Compute the optimal bid and ask depths for all inventory states q using
    stationary Value Iteration.
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
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # We want a very fine grid for depths
    max_depth = 3.0
    n_grid = 1000  # Much finer than 300
    grid = np.linspace(0, max_depth, n_grid)
    
    # Precompute probabilities
    exp_term = np.exp(-k * grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    one_minus_p = 1.0 - p_grid
    
    # Precompute cost for each state
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt  # (n_states,)
    
    # Precompute broadcast arrays
    grid_b = grid[None, :, None]   # (1, n_grid, 1)
    grid_a = grid[None, None, :]   # (1, 1, n_grid)
    pb_col = p_grid[None, :, None]         # (1, n_grid, 1)
    pa_row = p_grid[None, None, :]         # (1, 1, n_grid)
    ompb_col = one_minus_p[None, :, None]  # (1, n_grid, 1)
    ompa_row = one_minus_p[None, None, :]  # (1, 1, n_grid)
    
    # Initialize V
    V = np.zeros(n_states)
    
    opt_bid = np.zeros(n_states)
    opt_ask = np.zeros(n_states)
    
    # Value Iteration
    # We iterate until convergence or time limit
    # Since it's stationary, the policy should converge to a fixed point.
    # The number of iterations needed depends on the discount factor (implicit 1.0 here, 
    # but bounded horizon makes it effectively discounted by terminal condition).
    # However, since we are approximating the finite horizon with a stationary policy,
    # we should iterate enough to let the "shadow" of the terminal condition propagate?
    # Actually, for a finite horizon, the stationary policy is NOT the exact solution.
    # But the depth choice is often dominated by local inventory and volatility,
    # which are time-invariant in the stationary limit.
    # The main time-dependence is the "urgency" to flatten inventory near T.
    # We handle that in the lookup with a time-shift.
    
    # How many iterations?
    # Let's run as many as we can fit, or until change is small.
    max_iters = 5000
    tol = 1e-6
    
    for it in range(max_iters):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        v_q = V
        v_qp = np.zeros(n_states)
        v_qp[:-1] = V[1:]
        v_qm = np.zeros(n_states)
        v_qm[1:] = V[:-1]
        
        # Build 3D total: (n_states, n_grid, n_grid)
        t1 = v_q[:, None, None] * ompb_col * ompa_row
        t2 = v_qp[:, None, None] * pb_col * ompa_row + grid_b * pb_col * ompa_row
        t3 = v_qm[:, None, None] * ompb_col * pa_row + grid_a * ompb_col * pa_row
        t4 = v_q[:, None, None] * pb_col * pa_row + (grid_b + grid_a) * pb_col * pa_row
        
        total = t1 + t2 + t3 + t4 - cost_per_state[:, None, None]
        
        total_flat = total.reshape(n_states, -1)
        max_idx = np.argmax(total_flat, axis=1)
        i = max_idx // n_grid
        j = max_idx % n_grid
        
        new_V = total_flat[np.arange(n_states), max_idx]
        new_bid = grid[i]
        new_ask = grid[j]
        
        # Check convergence
        diff = np.max(np.abs(new_V - V))
        V = new_V
        opt_bid = new_bid
        opt_ask = new_ask
        
        if diff < tol:
            break
            
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed stationary policy,
    applying a time-dependent inventory shift.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_stationary_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
        
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Time-dependent shift:
    # As t approaches T, we want to flatten inventory more aggressively.
    # This is approximated by shifting the inventory state used for lookup.
    # If q > 0, we shift towards higher q (more aggressive bid reduction / ask increase?)
    # Actually, if we have positive inventory, we want to sell more (wider ask, tighter bid? No, tighter ask to sell faster).
    # In the stationary policy, the depth for q is fixed.
    # To simulate "urgency", we can pretend our inventory is larger than it is,
    # which would trigger a stronger inventory risk aversion response (wider spreads).
    # Or, we can shift the inventory index.
    
    # Let's use a simple linear decay of the "effective" inventory.
    # Effective inventory q_eff = q * (1 - alpha * t/T) ?
    # If q_eff < q, we are less worried about inventory, so we quote tighter (more competitive).
    # Wait, if we are worried about inventory, we widen spreads.
    # At t=0, we are less worried (long time to flatten). At t=T, we are most worried.
    # So we want the policy to behave like it has HIGHER inventory risk near T.
    # So q_eff should be LARGER than q near T.
    # q_eff = q * (1 + beta * t/T)
    
    # Let's try a shift that increases the effective inventory magnitude near T.
    # This will make the stationary policy (which is trained on fixed q) act more "defensive" (wider) near T.
    time_frac = t / T
    # Beta controls the strength of the time-dependence.
    # A value of 0.5 means at T, we act as if inventory is 1.5x larger.
    beta = 0.5
    q_eff = inventory * (1.0 + beta * time_frac)
    
    # Round to nearest integer for lookup
    q_idx = np.round(q_eff).astype(int)
    
    # Clip to valid range
    q_idx = np.clip(q_idx, -Q, Q)
    
    indices = q_idx + Q
    
    bids = bid_depths[indices]
    asks = ask_depths[indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
