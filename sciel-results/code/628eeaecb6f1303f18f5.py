"""
Market making policy using value iteration on a coarse inventory grid.
Approximates the optimal policy with an affine depth function d(q) = base + skew*q,
where base and skew are time-dependent coefficients learned via value iteration.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 290.0

# Cache for policy coefficients per regime
_POLICY_CACHE = {}

def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute time-dependent policy coefficients via value iteration.
    Returns arrays: base_coeffs[steps], skew_coeffs[steps]
    where depth(q, n) = base_coeffs[n] + skew_coeffs[n] * q
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    if _remaining_budget() < 5.0:
        # Fallback: simple heuristic
        base = np.full(steps, 1.0 / k)
        skew = np.zeros(steps)
        _POLICY_CACHE[key] = (base, skew)
        return _POLICY_CACHE[key]
    
    dt = T / steps
    max_depth = 3.0 / k
    min_depth = 1e-6
    
    # Coarse inventory grid for value iteration
    # Use a subset of inventory levels to keep computation tractable
    n_inv_levels = 21
    inv_grid = np.linspace(-Q, Q, n_inv_levels)
    
    # For each time step (backwards), compute optimal depth for each inventory level
    # Then fit an affine function to approximate the optimal policy
    
    base_coeffs = np.zeros(steps)
    skew_coeffs = np.zeros(steps)
    
    # We'll do value iteration for a limited number of steps from the end
    # For early steps, use extrapolation or a simpler model
    
    # Precompute fill probability and earned depth for a grid of depths
    n_depth_levels = 50
    depth_grid = np.linspace(min_depth, max_depth, n_depth_levels)
    
    # For each inventory level and each time step, find optimal depth
    # This is expensive, so we'll do it for a subset of time steps and interpolate
    
    # Number of time steps to solve explicitly
    n_solve_steps = min(steps, 100)
    
    # Value function at terminal time is 0 (no terminal charge)
    # V[q] = 0 for all q at t = T
    
    # We'll store V for each inventory level at each solved time step
    # V[i, j] = value at time step (steps - 1 - i) for inventory level j
    V = np.zeros((n_solve_steps + 1, n_inv_levels))
    
    for step_idx in range(n_solve_steps):
        if _remaining_budget() < 2.0:
            break
        t_remaining = (steps - 1 - step_idx) * dt
        n = steps - 1 - step_idx  # actual time step index
        
        for j, q in enumerate(inv_grid):
            best_val = -np.inf
            best_depth = 1.0 / k
            
            # Try each depth level
            for d in depth_grid:
                if d < min_depth or d > max_depth:
                    continue
                
                # Bid fill probability
                if q >= Q:
                    bid_prob = 0.0
                else:
                    bid_prob = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                
                # Ask fill probability
                if q <= -Q:
                    ask_prob = 0.0
                else:
                    ask_prob = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                
                # Expected immediate reward: depth earned - inventory penalty
                # Inventory penalty: (gamma/2) * sigma^2 * q^2 * dt
                inv_penalty = 0.5 * gamma * sigma**2 * q**2 * dt
                
                # Expected depth earned
                # If bid fills: earn d, inventory becomes q+1
                # If ask fills: earn d, inventory becomes q-1
                # Both can fill independently
                p_bid = bid_prob
                p_ask = ask_prob
                p_both = p_bid * p_ask
                p_bid_only = p_bid * (1 - p_ask)
                p_ask_only = p_ask * (1 - p_bid)
                p_none = (1 - p_bid) * (1 - p_ask)
                
                # Expected depth earned
                exp_depth = d * (p_bid + p_ask)
                
                # Next inventory distribution
                # q+1 with prob p_bid_only + p_both
                # q-1 with prob p_ask_only + p_both
                # q with prob p_none
                q_plus = min(Q, q + 1)
                q_minus = max(-Q, q - 1)
                
                # Find indices in inv_grid for q_plus and q_minus
                # Use nearest neighbor
                idx_plus = np.argmin(np.abs(inv_grid - q_plus))
                idx_minus = np.argmin(np.abs(inv_grid - q_minus))
                
                # Continuation value
                # V is stored backwards: V[step_idx+1, j] is the value at the next time step
                cont_val = (p_bid_only + p_both) * V[step_idx + 1, idx_plus] + \
                           (p_ask_only + p_both) * V[step_idx + 1, idx_minus] + \
                           p_none * V[step_idx + 1, j]
                
                total_val = exp_depth - inv_penalty + cont_val
                
                if total_val > best_val:
                    best_val = total_val
                    best_depth = d
            
            V[step_idx, j] = best_val
            base_coeffs[n] = best_depth
            # We'll fit skew later
        
        # Check budget
        if _remaining_budget() < 2.0:
            break
    
    # For steps not solved explicitly, use the last computed values
    # Fit skew by looking at how base_coeffs varies with inventory
    # Actually, base_coeffs above is for a specific inventory, so we need to redo
    
    # Simpler approach: for each time step, compute optimal depth for several inventory levels
    # and fit a line
    
    base_coeffs = np.zeros(steps)
    skew_coeffs = np.zeros(steps)
    
    n_inv_for_fit = 5
    inv_fit = np.linspace(-Q, Q, n_inv_for_fit)
    
    for step_idx in range(n_solve_steps):
        if _remaining_budget() < 1.0:
            break
        n = steps - 1 - step_idx
        
        depths_for_inv = np.zeros(n_inv_for_fit)
        
        for j, q in enumerate(inv_fit):
            best_val = -np.inf
            best_depth = 1.0 / k
            
            for d in depth_grid:
                if d < min_depth or d > max_depth:
                    continue
                
                if q >= Q:
                    bid_prob = 0.0
                else:
                    bid_prob = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                
                if q <= -Q:
                    ask_prob = 0.0
                else:
                    ask_prob = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                
                inv_penalty = 0.5 * gamma * sigma**2 * q**2 * dt
                
                p_bid = bid_prob
                p_ask = ask_prob
                p_both = p_bid * p_ask
                p_bid_only = p_bid * (1 - p_ask)
                p_ask_only = p_ask * (1 - p_bid)
                p_none = (1 - p_bid) * (1 - p_ask)
                
                exp_depth = d * (p_bid + p_ask)
                
                q_plus = min(Q, q + 1)
                q_minus = max(-Q, q - 1)
                
                idx_plus = np.argmin(np.abs(inv_grid - q_plus))
                idx_minus = np.argmin(np.abs(inv_grid - q_minus))
                
                cont_val = (p_bid_only + p_both) * V[step_idx + 1, idx_plus] + \
                           (p_ask_only + p_both) * V[step_idx + 1, idx_minus] + \
                           p_none * V[step_idx + 1, j]
                
                total_val = exp_depth - inv_penalty + cont_val
                
                if total_val > best_val:
                    best_val = total_val
                    best_depth = d
            
            depths_for_inv[j] = best_depth
        
        # Fit linear: depth = base + skew * q
        if n_inv_for_fit >= 2:
            A_mat = np.vstack([np.ones(n_inv_for_fit), inv_fit]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_mat, depths_for_inv, rcond=None)
                base_coeffs[n] = coeffs[0]
                skew_coeffs[n] = coeffs[1]
            except:
                base_coeffs[n] = np.mean(depths_for_inv)
                skew_coeffs[n] = 0.0
        else:
            base_coeffs[n] = depths_for_inv[0]
            skew_coeffs[n] = 0.0
    
    # For early steps (not solved), use the values from the earliest solved step
    # with time decay on skew
    if n_solve_steps < steps:
        # Find the earliest solved step
        earliest_solved = steps - n_solve_steps
        # For steps before earliest_solved, use extrapolation
        for n in range(earliest_solved):
            # Linear extrapolation or just use the earliest solved values
            base_coeffs[n] = base_coeffs[earliest_solved]
            # Decay skew for early steps
            time_frac = n / steps
            skew_coeffs[n] = skew_coeffs[earliest_solved] * (1.0 - 0.5 * time_frac)
    
    # Ensure depths are in valid range
    # We'll clip in the quote function
    
    _POLICY_CACHE[key] = (base_coeffs, skew_coeffs)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    if _remaining_budget() < 0.5:
        # Emergency fallback
        max_depth = 3.0 / k
        bid = np.full(len(inventory), 0.5 * max_depth)
        ask = np.full(len(inventory), 0.5 * max_depth)
        return bid, ask
    
    # Get policy coefficients
    base_coeffs, skew_coeffs = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current time step
    n = int(round(t / (T / steps)))
    n = max(0, min(steps - 1, n))
    
    inv = inventory.astype(np.float64)
    
    # Compute depths: d = base + skew * q
    base = base_coeffs[n]
    skew = skew_coeffs[n]
    
    # For bid: we want to be more aggressive when inventory is high (positive)
    # So bid depth should decrease with inventory: bid = base - skew * q
    # For ask: ask depth should increase with inventory: ask = base + skew * q
    # But we need to be careful about the sign convention
    
    # Let's define:
    # bid_depth = base + skew_bid * q
    # ask_depth = base + skew_ask * q
    # When q > 0 (long), we want to reduce bid (to avoid more long) and increase ask
    # So skew_bid should be negative, skew_ask should be positive
    
    # From our fit, depth = base + skew * q
    # For a symmetric policy, bid and ask should be mirror images
    # Let's use:
    # bid_depth = base - skew * q  (when q>0, bid decreases)
    # ask_depth = base + skew * q  (when q>0, ask increases)
    
    bid_depths = base - skew * inv
    ask_depths = base + skew * inv
    
    # Clip to valid range
    min_depth = 1e-6
    max_depth = 3.0 / k
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Handle inventory bounds
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
