"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for DP tables keyed by regime parameters
_dp_cache = {}
_start_time = time.time()
_TIME_LIMIT = 290  # seconds, leave margin before 330s kill

def _build_dp_table(sigma, A, k, gamma, T, steps, Q):
    """
    Build the optimal policy table via backward induction.
    Returns a dict mapping (step, inventory) -> (bid_depth, ask_depth).
    """
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # Candidate depths: use a grid of values
    # Fill prob = 1 - exp(-A * exp(-k*d) * dt)
    # We want to cover a range of fill probabilities
    # d = -log(-log(1-p)/A)/k for a given p
    # Use a grid of depths
    min_d = 1e-6
    max_d = 5.0 / k  # reasonably large depth
    # Use 40 candidate depths
    n_candidates = 40
    candidates = np.linspace(min_d, max_d, n_candidates)
    
    # V[n][q_idx] where q_idx = q + Q
    # V is (n_steps+1, n_inv)
    V = np.zeros((n_steps + 1, n_inv))
    # Policy: bid_depth[n][q_idx], ask_depth[n][q_idx]
    bid_policy = np.zeros((n_steps, n_inv))
    ask_policy = np.zeros((n_steps, n_inv))
    
    # Backward induction from n = n_steps-1 down to 0
    # At n = n_steps, V = 0 (no terminal charge)
    
    for n in range(n_steps - 1, -1, -1):
        # Check time
        if time.time() - _start_time > _TIME_LIMIT - 5:
            # Fallback: use simple heuristic
            for q_idx in range(n_inv):
                bid_policy[n, q_idx] = 1.0 / k
                ask_policy[n, q_idx] = 1.0 / k
            # Fill remaining
            for n2 in range(n - 1, -1, -1):
                for q_idx in range(n_inv):
                    bid_policy[n2, q_idx] = 1.0 / k
                    ask_policy[n2, q_idx] = 1.0 / k
            return bid_policy, ask_policy, V
        
        # Precompute exp(-k * candidates) * A * dt for fill probabilities
        # p = 1 - exp(-A * exp(-k*d) * dt)
        exp_term = np.exp(-k * candidates)  # (n_candidates,)
        p_candidates = 1.0 - np.exp(-A * exp_term * dt)  # (n_candidates,)
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory penalty for this step
            inv_penalty = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # If q == Q, no bid fill possible
            # If q == -Q, no ask fill possible
            
            best_val = -np.inf
            best_bid = 1.0 / k
            best_ask = 1.0 / k
            
            # Try all combinations of bid and ask depths
            # To keep it tractable, we can use a reduced grid or optimize
            # For now, use full grid but with early exit if time is tight
            for i in range(n_candidates):
                bid_d = candidates[i]
                p_bid = p_candidates[i]
                
                # If q == Q, bid fill not possible, so p_bid effectively 0
                if q == Q:
                    p_bid = 0.0
                
                for j in range(n_candidates):
                    ask_d = candidates[j]
                    p_ask = p_candidates[j]
                    
                    if q == -Q:
                        p_ask = 0.0
                    
                    # Expected future value
                    # After fills:
                    # - bid fill: q -> q-1
                    # - ask fill: q -> q+1
                    # - no fill: q -> q
                    # - both: q -> q (bid then ask, net 0)
                    
                    # Probabilities:
                    p_both = p_bid * p_ask
                    p_bid_only = p_bid * (1 - p_ask)
                    p_ask_only = (1 - p_bid) * p_ask
                    p_none = (1 - p_bid) * (1 - p_ask)
                    
                    # Future values
                    v_bid_only = V[n + 1, q_idx - 1] if q - 1 >= -Q else V[n + 1, q_idx]
                    v_ask_only = V[n + 1, q_idx + 1] if q + 1 <= Q else V[n + 1, q_idx]
                    v_none = V[n + 1, q_idx]
                    v_both = V[n + 1, q_idx]  # net inventory change 0
                    
                    # Immediate reward: depth earned minus penalty
                    # Bid fill earns -bid_depth (we pay the depth)
                    # Ask fill earns +ask_depth (we receive the depth)
                    # Wait: "A fill earns its depth" - bid fill: we sell at bid, so we earn (mid - bid_depth) vs mid, so we lose bid_depth relative to mid
                    # Actually: "depths earned" - for a bid fill, we execute at bid = mid - bid_depth, so we earn -bid_depth relative to mid
                    # For an ask fill, we execute at ask = mid + ask_depth, so we earn +ask_depth relative to mid
                    
                    immediate = -p_bid * bid_d + p_ask * ask_d - inv_penalty
                    
                    future = p_bid_only * v_bid_only + p_ask_only * v_ask_only + p_both * v_both + p_none * v_none
                    
                    total = immediate + future
                    
                    if total > best_val:
                        best_val = total
                        best_bid = bid_d
                        best_ask = ask_d
            
            V[n, q_idx] = best_val
            bid_policy[n, q_idx] = best_bid
            ask_policy[n, q_idx] = best_ask
    
    return bid_policy, ask_policy, V


def _get_dp_tables(params):
    """Get or build DP tables for the given regime."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _dp_cache:
        _dp_cache[key] = _build_dp_table(sigma, A, k, gamma, T, steps, Q)
    
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    """
    try:
        # Check time
        if time.time() - _start_time > _TIME_LIMIT - 2:
            # Fallback: simple heuristic
            k = params['k']
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
        
        bid_policy, ask_policy, _ = _get_dp_tables(params)
        
        # t = n * dt, so n = t / dt
        dt = params['T'] / params['steps']
        n = int(round(t / dt))
        
        # Clamp n to valid range
        n = max(0, min(n, params['steps'] - 1))
        
        Q = params['Q']
        
        # inventory is in [-Q, Q], map to index
        inv_idx = inventory + Q
        inv_idx = np.clip(inv_idx, 0, 2 * Q)
        
        bid_depths = bid_policy[n, inv_idx]
        ask_depths = ask_policy[n, inv_idx]
        
        # Ensure no negative depths and no NaN
        bid_depths = np.maximum(bid_depths, 1e-6)
        ask_depths = np.maximum(ask_depths, 1e-6)
        
        return bid_depths, ask_depths
    except Exception:
        # Fallback on any error
        k = params['k']
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
# EVOLVE-BLOCK-END

