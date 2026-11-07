"""
Market making policy using backward induction with a quadratic value function approximation.
Uses V(q,t) = a(t)*q^2 + b(t)*q + c(t) to capture inventory risk and asymmetry.
The optimal depth is found by maximizing immediate reward + expected continuation value.
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
    Compute time-dependent policy using backward induction with quadratic value function.
    Returns arrays: bid_base[steps], bid_skew[steps], ask_base[steps], ask_skew[steps]
    where bid_depth(q,n) = bid_base[n] + bid_skew[n]*q
    and ask_depth(q,n) = ask_base[n] + ask_skew[n]*q
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    if _remaining_budget() < 5.0:
        # Fallback: simple heuristic
        base = np.full(steps, 1.0 / k)
        skew = np.zeros(steps)
        _POLICY_CACHE[key] = (base, -skew, base, skew)
        return _POLICY_CACHE[key]
    
    dt = T / steps
    max_depth = 3.0 / k
    min_depth = 1e-6
    
    # Use a finer depth grid for better optimization
    n_depth_levels = 80
    depth_grid = np.linspace(min_depth, max_depth, n_depth_levels)
    
    # Number of inventory levels for value function fitting
    n_inv_levels = 61
    inv_grid = np.linspace(-Q, Q, n_inv_levels)
    
    # Value function coefficients: V(q) = a*q^2 + b*q + c
    # At terminal time, V = 0
    a_coeffs = np.zeros(steps)  # quadratic coefficient
    b_coeffs = np.zeros(steps)  # linear coefficient  
    c_coeffs = np.zeros(steps)  # constant coefficient
    
    # Policy coefficients
    bid_base = np.zeros(steps)
    bid_skew = np.zeros(steps)
    ask_base = np.zeros(steps)
    ask_skew = np.zeros(steps)
    
    # We'll solve backward from the last step
    # For each step, compute optimal bid and ask depths for each inventory level
    # Then fit the value function and policy
    
    # Precompute fill probabilities for the depth grid
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Solve for all steps (or as many as budget allows)
    n_solve = min(steps, 200)  # Cap at 200 steps for safety
    
    for step_idx in range(n_solve):
        if _remaining_budget() < 2.0:
            break
        
        n = steps - 1 - step_idx  # actual time step index (going backward)
        
        # Current value function coefficients (from next step)
        if step_idx == 0:
            a_next = 0.0
            b_next = 0.0
            c_next = 0.0
        else:
            a_next = a_coeffs[n + 1] if n + 1 < steps else 0.0
            b_next = b_coeffs[n + 1] if n + 1 < steps else 0.0
            c_next = c_coeffs[n + 1] if n + 1 < steps else 0.0
        
        # For each inventory level, find optimal bid and ask depths
        # We'll store optimal depths for later fitting
        opt_bid_depths = np.zeros(n_inv_levels)
        opt_ask_depths = np.zeros(n_inv_levels)
        opt_values = np.zeros(n_inv_levels)
        
        for j, q in enumerate(inv_grid):
            # Inventory penalty for this step
            inv_penalty = 0.5 * gamma * sigma**2 * q**2 * dt
            
            best_bid_val = -np.inf
            best_ask_val = -np.inf
            best_bid_d = 1.0 / k
            best_ask_d = 1.0 / k
            
            # Try each depth for bid
            for di, d in enumerate(depth_grid):
                if q >= Q:
                    continue  # No bid fill at max inventory
                
                p_bid = fill_probs[di]
                
                # Expected immediate reward from bid
                # If bid fills: earn d, inventory becomes q+1
                # If not: earn 0, inventory stays q
                # We need to consider the ask side too, but for simplicity
                # in this first pass, we'll handle bid and ask separately
                # and combine them
                
                # For now, let's compute the value if we quote bid at depth d
                # and ask at some depth (we'll optimize jointly below)
                pass
            
            # Joint optimization: try combinations of bid and ask depths
            # This is expensive, so we'll use a smarter approach
            # Assume symmetric policy: bid_depth = base_b - skew_b*q, ask_depth = base_a + skew_a*q
            # For now, let's just find the best single depth for both
            
            best_total_val = -np.inf
            best_d = 1.0 / k
            
            for di, d in enumerate(depth_grid):
                p = fill_probs[di]
                
                # Bid fill probability
                if q >= Q:
                    p_bid = 0.0
                else:
                    p_bid = p
                
                # Ask fill probability
                if q <= -Q:
                    p_ask = 0.0
                else:
                    p_ask = p
                
                p_both = p_bid * p_ask
                p_bid_only = p_bid * (1 - p_ask)
                p_ask_only = p_ask * (1 - p_bid)
                p_none = (1 - p_bid) * (1 - p_ask)
                
                # Expected depth earned
                exp_depth = d * (p_bid + p_ask)
                
                # Next inventory distribution
                q_plus = min(Q, q + 1)
                q_minus = max(-Q, q - 1)
                
                # Continuation value using quadratic approximation
                # V(q+1) = a_next*(q+1)^2 + b_next*(q+1) + c_next
                # V(q-1) = a_next*(q-1)^2 + b_next*(q-1) + c_next
                # V(q) = a_next*q^2 + b_next*q + c_next
                
                v_plus = a_next * q_plus**2 + b_next * q_plus + c_next
                v_minus = a_next * q_minus**2 + b_next * q_minus + c_next
                v_same = a_next * q**2 + b_next * q + c_next
                
                cont_val = (p_bid_only + p_both) * v_plus + \
                           (p_ask_only + p_both) * v_minus + \
                           p_none * v_same
                
                total_val = exp_depth - inv_penalty + cont_val
                
                if total_val > best_total_val:
                    best_total_val = total_val
                    best_d = d
            
            opt_bid_depths[j] = best_d
            opt_ask_depths[j] = best_d
            opt_values[j] = best_total_val
        
        # Fit value function: V(q) = a*q^2 + b*q + c
        A_mat = np.vstack([inv_grid**2, inv_grid, np.ones(n_inv_levels)]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, opt_values, rcond=None)
            a_coeffs[n] = coeffs[0]
            b_coeffs[n] = coeffs[1]
            c_coeffs[n] = coeffs[2]
        except:
            a_coeffs[n] = 0.5 * gamma * sigma**2 * dt
            b_coeffs[n] = 0.0
            c_coeffs[n] = 0.0
        
        # Fit policy: bid_depth = bid_base + bid_skew*q
        # For bid: when q > 0 (long), we want smaller bid depth (wider spread)
        # So bid_skew should be negative
        # When q < 0 (short), we want larger bid depth (narrower spread)
        
        # Fit linear to opt_bid_depths
        A_bid = np.vstack([np.ones(n_inv_levels), inv_grid]).T
        try:
            coeffs_bid, _, _, _ = np.linalg.lstsq(A_bid, opt_bid_depths, rcond=None)
            bid_base[n] = coeffs_bid[0]
            bid_skew[n] = coeffs_bid[1]
        except:
            bid_base[n] = np.mean(opt_bid_depths)
            bid_skew[n] = 0.0
        
        # Fit linear to opt_ask_depths
        # For ask: when q > 0 (long), we want larger ask depth (narrower spread)
        # So ask_skew should be positive
        A_ask = np.vstack([np.ones(n_inv_levels), inv_grid]).T
        try:
            coeffs_ask, _, _, _ = np.linalg.lstsq(A_ask, opt_ask_depths, rcond=None)
            ask_base[n] = coeffs_ask[0]
            ask_skew[n] = coeffs_ask[1]
        except:
            ask_base[n] = np.mean(opt_ask_depths)
            ask_skew[n] = 0.0
    
    # For early steps not solved, use the earliest solved values
    if n_solve < steps:
        earliest = steps - n_solve
        for n in range(earliest):
            time_frac = n / steps
            # Decay skew for early steps (less urgent to hedge)
            decay = 1.0 - 0.3 * time_frac
            bid_base[n] = bid_base[earliest]
            bid_skew[n] = bid_skew[earliest] * decay
            ask_base[n] = ask_base[earliest]
            ask_skew[n] = ask_skew[earliest] * decay
    
    # Ensure depths are in valid range
    min_depth = 1e-6
    max_depth = 3.0 / k
    
    _POLICY_CACHE[key] = (bid_base, bid_skew, ask_base, ask_skew)
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
    bid_base, bid_skew, ask_base, ask_skew = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current time step
    n = int(round(t / (T / steps)))
    n = max(0, min(steps - 1, n))
    
    inv = inventory.astype(np.float64)
    
    # Compute depths
    bid_depths = bid_base[n] + bid_skew[n] * inv
    ask_depths = ask_base[n] + ask_skew[n] * inv
    
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
