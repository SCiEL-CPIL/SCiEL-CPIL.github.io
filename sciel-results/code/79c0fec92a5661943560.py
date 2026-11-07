"""Market-making with inventory risk: HJB-based policy via value iteration on inventory grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed policies
_policy_cache = {}
_start_time = None


def _get_remaining_budget():
    """Return seconds remaining before the 300s kill, with safety margin."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 300.0 - elapsed - 5.0)  # 5s safety margin


def _solve_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete-time optimal market-making problem via backward induction.
    
    State: inventory q in [-Q, Q], time step n in [0, steps).
    Action: choose bid_depth d_b >= 0 and ask_depth d_a >= 0 (or inf for no quote).
    
    The HJB for this discrete model:
    V(n, q) = max over (d_b, d_a) of [
        immediate_reward(n, q, d_b, d_a) + E[V(n+1, q')]
    ]
    
    where immediate reward includes the depth earned from fills minus the holding cost.
    """
    dt = T / steps
    n_states = 2 * Q + 1
    
    # V[n, q] for q in 0..2Q representing inventory q-Q
    # V has shape (steps+1, n_states)
    V = np.zeros((steps + 1, n_states), dtype=np.float64)
    
    # Policy: store optimal depths
    # policy_bid[n, q] = optimal bid depth at step n, inventory q-Q
    # policy_ask[n, q] = optimal ask depth
    policy_bid = np.full((steps, n_states), 1.0 / k, dtype=np.float64)
    policy_ask = np.full((steps, n_states), 1.0 / k, dtype=np.float64)
    
    # For each step, we need to optimize over bid and ask depths.
    # Since bid and ask are coupled only through the next state (inventory change),
    # and the fill probabilities depend on depth, we can do a joint optimization.
    
    # Key insight: the optimal depth for each side can be found by considering
    # the marginal benefit vs cost. Let's use a grid search over depths.
    
    # Depth grid: from 0 to some max_depth
    max_depth = 5.0 / k  # reasonable upper bound
    n_depth_points = 50
    depth_grid = np.linspace(0, max_depth, n_depth_points)
    
    # Precompute fill probabilities for each depth
    # P_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if _get_remaining_budget() < 2.0:
            # Not enough time; use heuristic for remaining steps
            for q_idx in range(n_states):
                q = q_idx - Q
                # Simple inventory adjustment
                adj = 0.05 * q / Q
                policy_bid[n, q_idx] = max(0.0, 1.0/k * (1.0 - adj))
                policy_ask[n, q_idx] = max(0.0, 1.0/k * (1.0 + adj))
            break
        
        for q_idx in range(n_states):
            q = q_idx - Q
            
            # Holding cost for this step
            hold_cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            best_value = -np.inf
            best_bid = 1.0 / k
            best_ask = 1.0 / k
            
            # Try combinations of bid and ask depths
            # To speed up, we can do a coarse-to-fine search
            for i_b in range(n_depth_points):
                d_b = depth_grid[i_b]
                p_b = fill_probs[i_b]
                
                # If q == Q, no bid fill possible
                if q == Q:
                    p_b = 0.0
                
                for i_a in range(n_depth_points):
                    d_a = depth_grid[i_a]
                    p_a = fill_probs[i_a]
                    
                    # If q == -Q, no ask fill possible
                    if q == -Q:
                        p_a = 0.0
                    
                    # Expected next value
                    # Fill scenarios:
                    # 1. No fill: prob (1-p_b)(1-p_a), q unchanged
                    # 2. Bid fill only: prob p_b(1-p_a), q -> q-1
                    # 3. Ask fill only: prob (1-p_b)p_a, q -> q+1
                    # 4. Both fill: prob p_b*p_a, q -> q (net zero)
                    
                    q_next_00 = q  # no fill
                    q_next_10 = q - 1  # bid fill
                    q_next_01 = q + 1  # ask fill
                    q_next_11 = q  # both fill
                    
                    # Clamp to valid range
                    q_next_10 = max(-Q, q_next_10)
                    q_next_01 = min(Q, q_next_01)
                    
                    idx_00 = q_next_00 + Q
                    idx_10 = q_next_10 + Q
                    idx_01 = q_next_01 + Q
                    idx_11 = q_next_11 + Q
                    
                    V_next = (
                        (1 - p_b) * (1 - p_a) * V[n + 1, idx_00] +
                        p_b * (1 - p_a) * V[n + 1, idx_10] +
                        (1 - p_b) * p_a * V[n + 1, idx_01] +
                        p_b * p_a * V[n + 1, idx_11]
                    )
                    
                    # Immediate reward: depth earned from fills
                    # Bid fill earns d_b, ask fill earns d_a
                    reward = p_b * d_b + p_a * d_a - hold_cost
                    
                    total = reward + V_next
                    
                    if total > best_value:
                        best_value = total
                        best_bid = d_b
                        best_ask = d_a
            
            V[n, q_idx] = best_value
            policy_bid[n, q_idx] = best_bid
            policy_ask[n, q_idx] = best_ask
    
    return policy_bid, policy_ask


def _get_policy(params):
    """Get or compute the policy for given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key in _policy_cache:
        return _policy_cache[key]
    
    policy_bid, policy_ask = _solve_policy(
        params['sigma'], params['A'], params['k'], 
        params['gamma'], params['T'], params['steps'], params['Q']
    )
    
    _policy_cache[key] = (policy_bid, policy_ask)
    return policy_bid, policy_ask


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventories in [-Q, Q]
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        Tuple of (bid_depths, ask_depths), each of shape (4000,)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check if we have time
    remaining = _get_remaining_budget()
    if remaining < 1.0:
        # Emergency fallback: simple heuristic
        Q = params['Q']
        k = params['k']
        base = 1.0 / k
        # Adjust for inventory: shift quotes to reduce inventory
        adj = 0.1 * inventory / Q
        bid_depths = np.maximum(0.0, base * (1.0 - adj))
        ask_depths = np.maximum(0.0, base * (1.0 + adj))
        return bid_depths, ask_depths
    
    # Determine current step
    T = params['T']
    steps = params['steps']
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    Q = params['Q']
    k = params['k']
    
    # Get policy
    try:
        policy_bid, policy_ask = _get_policy(params)
    except Exception:
        # Fallback to simple heuristic
        base = 1.0 / k
        adj = 0.1 * inventory / Q
        bid_depths = np.maximum(0.0, base * (1.0 - adj))
        ask_depths = np.maximum(0.0, base * (1.0 + adj))
        return bid_depths, ask_depths
    
    # Map inventory to policy index
    # inventory is in [-Q, Q], policy index is inventory + Q
    inv_idx = np.clip(inventory, -Q, Q).astype(int) + Q
    
    # Look up depths
    bid_depths = policy_bid[n, inv_idx].astype(np.float64)
    ask_depths = policy_ask[n, inv_idx].astype(np.float64)
    
    # Ensure validity: depths must be >= -3/k (which is always true for non-negative)
    # and not NaN or -inf
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, 1.0/k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, 1.0/k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
