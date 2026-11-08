import numpy as np
import time

# Global cache for pre-computed DP policies
_policy_cache = {}
_DEADLINE = time.time() + 325.0  # 325s wall-clock deadline (5s safety margin)


def _compute_dp_policy(params, depth_grid):
    """
    Compute optimal bid/ask depth indices for each step and inventory level
    via backward induction (dynamic programming).
    
    Returns:
        policy_bid: array of shape (steps, 2*Q+1) with indices into depth_grid
        policy_ask: array of shape (steps, 2*Q+1) with indices into depth_grid
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
    
    # Precompute fill probabilities and depths earned for each depth grid point
    # depth_grid[i] is the depth, and we precompute:
    #   prob[i] = 1 - exp(-A * exp(-k * depth[i]) * dt)
    #   depth[i] is the depth earned on fill
    depths = depth_grid
    probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Inventory penalty per step: (gamma/2) * sigma^2 * q^2 * dt
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    inv_penalty = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # Value function: V[t, q_idx] where q_idx = q + Q
    V_next = np.zeros(n_states, dtype=np.float64)
    
    # Policy arrays
    policy_bid = np.zeros((steps, n_states), dtype=np.int32)
    policy_ask = np.zeros((steps, n_states), dtype=np.int32)
    
    # Markers: we use -1 for "no quote" (inf depth)
    # We'll store the index into depth_grid, or a special value for no-quote
    # Let's use n_depths as the "no quote" index
    n_depths = len(depths)
    policy_bid[:] = n_depths  # default: no quote
    policy_ask[:] = n_depths  # default: no quote
    
    for t in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            break
        
        V_curr = np.zeros(n_states, dtype=np.float64)
        
        for q_idx in range(n_states):
            q = q_idx - Q
            
            # Try all combinations of bid depth index and ask depth index
            # bid_idx in [0, n_depths) or n_depths (no quote)
            # ask_idx in [0, n_depths) or n_depths (no quote)
            
            # For efficiency, precompute the contribution of each depth choice
            # For a given bid depth index bi:
            #   if bi == n_depths: no bid quote, prob=0, depth_earned=0, inv_change=0
            #   else: prob=probs[bi], depth_earned=depths[bi], inv_change=+1 (if q < Q)
            # Similarly for ask.
            
            # Bid options: indices 0..n_depths-1 (quote) or n_depths (no quote)
            # But if q >= Q, no bid fill allowed, so bid must be no-quote
            # If q < Q, bid can be any depth or no-quote
            
            # Ask options: indices 0..n_depths-1 (quote) or n_depths (no quote)
            # But if q <= -Q, no ask fill allowed, so ask must be no-quote
            # If q > -Q, ask can be any depth or no-quote
            
            best_val = -np.inf
            best_bi = n_depths
            best_ai = n_depths
            
            # Determine valid bid indices
            if q >= Q:
                valid_bi = [n_depths]  # only no-quote
            else:
                valid_bi = list(range(n_depths + 1))  # 0..n_depths-1 and n_depths
            
            if q <= -Q:
                valid_ai = [n_depths]  # only no-quote
            else:
                valid_ai = list(range(n_depths + 1))
            
            for bi in valid_bi:
                # Bid fill
                if bi == n_depths:
                    p_b = 0.0
                    d_b = 0.0
                    inv_b = q  # no change
                else:
                    p_b = probs[bi]
                    d_b = depths[bi]
                    inv_b = q + 1  # bid fill increases inventory
            
            for ai in valid_ai:
                # Ask fill
                if ai == n_depths:
                    p_a = 0.0
                    d_a = 0.0
                    inv_a = q  # no change
                else:
                    p_a = probs[ai]
                    d_a = depths[ai]
                    inv_a = q - 1  # ask fill decreases inventory
            
            # Now compute expected value for this (bi, ai) combination
            # Possible outcomes:
            #   bid only: prob p_b * (1 - p_a), inv = q+1, reward = d_b
            #   ask only: prob (1 - p_b) * p_a, inv = q-1, reward = d_a
            #   both:     prob p_b * p_a, inv = q, reward = d_b + d_a
            #   neither:  prob (1 - p_b) * (1 - p_a), inv = q, reward = 0
            
            # Current step reward (depths earned) minus inventory penalty
            # V = immediate_reward - inv_penalty + E[V_next]
            
            # Let's compute E[V_next] for each outcome
            # inv index for q+1: (q+1) + Q = q_idx + 1 (if q < Q)
            # inv index for q-1: (q-1) + Q = q_idx - 1 (if q > -Q)
            # inv index for q: q_idx
            
            val_bid_only = 0.0
            val_ask_only = 0.0
            val_both = 0.0
            val_neither = 0.0
            
            if p_b * (1 - p_a) > 0:
                new_q_idx = q_idx + 1
                if 0 <= new_q_idx < n_states:
                    val_bid_only = d_b - inv_penalty[q_idx] + p_b * (1 - p_a) * V_next[new_q_idx]
                else:
                    val_bid_only = d_b - inv_penalty[q_idx]
            # Wait, I need to be more careful. The reward is earned when fill happens.
            # Let me recompute:
            # Expected immediate reward = p_b*(1-p_a)*d_b + (1-p_b)*p_a*d_a + p_b*p_a*(d_b+d_a)
            # = p_b*d_b + p_a*d_a  (since p_b*(1-p_a)*d_b + p_b*p_a*d_b = p_b*d_b)
            
            # So expected immediate reward is simply p_b * d_b + p_a * d_a
            # And the inventory penalty is inv_penalty[q_idx] (based on current q before fills)
            
            # E[V_next] = p_b*(1-p_a)*V_next[q_idx+1] + (1-p_b)*p_a*V_next[q_idx-1] + (p_b*p_a + (1-p_b)*(1-p_a))*V_next[q_idx]
            
            exp_reward = p_b * d_b + p_a * d_a
            exp_penalty = inv_penalty[q_idx]
            
            # E[V_next]
            e_v = 0.0
            # bid only: q+1
            if p_b * (1 - p_a) > 0 and q_idx + 1 < n_states:
                e_v += p_b * (1 - p_a) * V_next[q_idx + 1]
            # ask only: q-1
            if (1 - p_b) * p_a > 0 and q_idx - 1 >= 0:
                e_v += (1 - p_b) * p_a * V_next[q_idx - 1]
            # both or neither: q
            p_stay = p_b * p_a + (1 - p_b) * (1 - p_a)
            if p_stay > 0:
                e_v += p_stay * V_next[q_idx]
            
            total_val = exp_reward - exp_penalty + e_v
            
            if total_val > best_val:
                best_val = total_val
                best_bi = bi
                best_ai = ai
            
            if time.time() > _DEADLINE:
                break
        
        V_curr[q_idx] = best_val
        policy_bid[t, q_idx] = best_bi
        policy_ask[t, q_idx] = best_ai
        
        if time.time() > _DEADLINE:
            break
        
        V_next = V_curr
    
    return policy_bid, policy_ask


def _heuristic_depths(inventory, params):
    """Fallback heuristic if DP not available."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    base = 1.0 / k
    inv = inventory.astype(np.float64)
    
    bid_depth = base + gamma * sigma**2 * (inv + 0.5) * dt
    ask_depth = base - gamma * sigma**2 * (inv - 0.5) * dt
    
    bid_depth = np.where(bid_depth > 0, bid_depth, np.inf)
    ask_depth = np.where(ask_depth > 0, ask_depth, np.inf)
    
    bid_depth = np.where(inv >= Q, np.inf, bid_depth)
    ask_depth = np.where(inv <= -Q, np.inf, ask_depth)
    
    return bid_depth, ask_depth


def quote(t, inventory, mid, params):
    """
    Quote using pre-computed DP policy with deadline guard.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        # Check if we have time to compute DP
        if time.time() > _DEADLINE - 50:  # Need at least 50s for DP + some slack
            # Fall back to heuristic
            return _heuristic_depths(inventory, params)
        
        # Create depth grid: log-spaced from small to large
        # Minimum depth: very small (near 0), maximum: maybe 5/k to 10/k
        k = params['k']
        min_d = 0.01 / k
        max_d = 10.0 / k
        n_depths = 100
        depth_grid = np.geomspace(min_d, max_d, n_depths)
        
        try:
            policy_bid, policy_ask = _compute_dp_policy(params, depth_grid)
            _policy_cache[key] = {
                'depths': depth_grid,
                'policy_bid': policy_bid,
                'policy_ask': policy_ask,
            }
        except Exception:
            # Fallback to heuristic
            return _heuristic_depths(inventory, params)
    
    policy = _policy_cache[key]
    depths = policy['depths']
    policy_bid = policy['policy_bid']
    policy_ask = policy['policy_ask']
    n_depths = len(depths)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step index
    dt = params['T'] / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    
    # Map inventory to index
    inv_idx = inventory + Q
    
    # Get policy indices
    bid_idx = policy_bid[n, inv_idx]
    ask_idx = policy_ask[n, inv_idx]
    
    # Map indices to depths; n_depths means no quote (inf)
    bid_depths = np.where(bid_idx < n_depths, depths[bid_idx], np.inf)
    ask_depths = np.where(ask_idx < n_depths, depths[ask_idx], np.inf)
    
    return bid_depths, ask_depths
