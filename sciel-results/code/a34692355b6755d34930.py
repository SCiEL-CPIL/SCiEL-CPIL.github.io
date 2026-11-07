"""Market making with inventory risk: structural DP approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for computed DP policies per regime
_dp_cache = {}
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 30.0  # seconds before kill to stop computing


def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _start_time)


def _compute_dp_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth policy via backward induction DP.
    
    State: (n, q) where n = step index, q = inventory in [-Q, Q]
    Action: choose bid_depth d_b >= 0, ask_depth d_a >= 0
    Value: expected remaining reward from state (n, q)
    
    Reward per step: depth_earned - (gamma/2)*sigma^2*q^2*dt
    - If bid hit: earn d_b, inventory goes q+1
    - If ask hit: earn d_a, inventory goes q-1
    - Independent, so both can happen (prob = p_b * p_a)
    
    Depth grid: use a fine grid in [0, d_max] where d_max is chosen to cover
    the region where fill probabilities are meaningful.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_states = 2 * Q + 1
    
    # Choose depth grid
    # Fill prob = 1 - exp(-A * exp(-k*d) * dt)
    # For small k*d, fill prob ~ A*exp(-k*d)*dt
    # We want to cover depths where fill prob is not negligible
    # d_max: where A*exp(-k*d_max)*dt is very small
    # Let's use d_max = max(1.0/k * 5, 1.0) but cap for safety
    d_max = min(5.0 / k, 3.0)
    
    # Use a non-uniform grid: fine near 0, coarser at larger depths
    n_depths = 60
    # Geometric spacing near 0, linear after
    # Points: 0, then geometric up to d_geo, then linear to d_max
    d_geo = 1.0 / k
    n_geo = 25
    d_grid = np.concatenate([
        np.geomspace(1e-4, d_geo, n_geo),
        np.linspace(d_geo, d_max, n_depths - n_geo)
    ])
    d_grid = np.clip(d_grid, 0.0, d_max)
    # Ensure d_grid[0] is small but positive
    d_grid = np.maximum(d_grid, 1e-6)
    n_d = len(d_grid)
    
    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * d_grid)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)
    # Shape: (n_d,)
    
    # Value function: V[n, q] for n from 0 to steps
    # V[steps, q] = 0 for all q (no terminal charge)
    V_next = np.zeros(n_states)
    
    # Policy storage: for each (n, q), store optimal bid/ask depth index
    # We'll store the full policy table
    policy_bid = np.zeros((steps, n_states), dtype=np.float64)
    policy_ask = np.zeros((steps, n_states), dtype=np.float64)
    
    for n in range(steps - 1, -1, -1):
        if _remaining_time() < 10.0:
            # Not enough time, use heuristic for remaining steps
            for q_idx in range(n_states):
                policy_bid[n, q_idx] = 1.0 / k
                policy_ask[n, q_idx] = 1.0 / k
            continue
        
        V_curr = np.zeros(n_states)
        # Inventory holding cost per step
        q_vals_arr = q_vals  # shape (n_states,)
        holding_cost = 0.5 * gamma * sigma**2 * q_vals_arr**2 * dt
        
        for q_idx, q in enumerate(q_vals):
            # Determine valid actions based on inventory bounds
            # No bid fill at q=Q (can't go above Q)
            # No ask fill at q=-Q (can't go below -Q)
            if q >= Q:
                # Can only have ask fills (or neither)
                # Bid depth doesn't matter (no bid fill), set to 0
                best_val = -np.inf
                best_d_a_idx = 0
                for da_idx in range(n_d):
                    p_a = p_fill[da_idx]
                    # No bid fill
                    p_b = 0.0
                    # Outcomes:
                    # Neither: prob (1-p_a), reward 0, next state q
                    # Ask hit: prob p_a, reward d_a, next state q-1
                    val = 0.0
                    val += (1 - p_a) * V_next[q_idx]
                    val += p_a * (d_grid[da_idx] + V_next[q_idx - 1])  # q-1 is q_idx-1
                    val -= holding_cost[q_idx]
                    if val > best_val:
                        best_val = val
                        best_d_a_idx = da_idx
                V_curr[q_idx] = best_val
                policy_bid[n, q_idx] = 0.0  # no bid
                policy_ask[n, q_idx] = d_grid[best_d_a_idx]
            elif q <= -Q:
                # Can only have bid fills
                best_val = -np.inf
                best_d_b_idx = 0
                for db_idx in range(n_d):
                    p_b = p_fill[db_idx]
                    p_a = 0.0
                    val = 0.0
                    val += (1 - p_b) * V_next[q_idx]
                    val += p_b * (d_grid[db_idx] + V_next[q_idx + 1])  # q+1 is q_idx+1
                    val -= holding_cost[q_idx]
                    if val > best_val:
                        best_val = val
                        best_d_b_idx = db_idx
                V_curr[q_idx] = best_val
                policy_bid[n, q_idx] = d_grid[best_d_b_idx]
                policy_ask[n, q_idx] = 0.0
            else:
                # Both sides possible
                best_val = -np.inf
                best_db_idx = 0
                best_da_idx = 0
                for db_idx in range(n_d):
                    p_b = p_fill[db_idx]
                    for da_idx in range(n_d):
                        p_a = p_fill[da_idx]
                        # Outcomes:
                        # Neither: (1-p_b)(1-p_a), reward 0, state q
                        # Bid only: p_b(1-p_a), reward d_b, state q+1
                        # Ask only: (1-p_b)p_a, reward d_a, state q-1
                        # Both: p_b*p_a, reward d_b+d_a, state q (q+1-1=q)
                        val = 0.0
                        val += (1 - p_b) * (1 - p_a) * V_next[q_idx]
                        val += p_b * (1 - p_a) * (d_grid[db_idx] + V_next[q_idx + 1])
                        val += (1 - p_b) * p_a * (d_grid[da_idx] + V_next[q_idx - 1])
                        val += p_b * p_a * (d_grid[db_idx] + d_grid[da_idx] + V_next[q_idx])
                        val -= holding_cost[q_idx]
                        if val > best_val:
                            best_val = val
                            best_db_idx = db_idx
                            best_da_idx = da_idx
                V_curr[q_idx] = best_val
                policy_bid[n, q_idx] = d_grid[best_db_idx]
                policy_ask[n, q_idx] = d_grid[best_da_idx]
        
        V_next = V_curr
    
    return policy_bid, policy_ask, d_grid


def quote(t, inventory, mid, params):
    """
    Quote bid/ask depths for each path based on DP-computed optimal policy.
    Falls back to heuristic if DP not yet available or time is short.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    
    # Cache key
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key in _dp_cache:
        policy_bid, policy_ask, d_grid = _dp_cache[cache_key]
    else:
        # Try to compute DP if we have time
        if _remaining_time() > 5.0:
            try:
                policy_bid, policy_ask, d_grid = _compute_dp_policy(sigma, A, k, gamma, T, steps, Q)
                _dp_cache[cache_key] = (policy_bid, policy_ask, d_grid)
            except Exception:
                # Fallback to heuristic
                depth = np.full(n_paths, 1.0 / k)
                return depth, depth.copy()
        else:
            # Not enough time, use heuristic
            depth = np.full(n_paths, 1.0 / k)
            return depth, depth.copy()
    
    # Lookup policy for each path
    q_vals = np.arange(-Q, Q + 1)
    q_to_idx = {q: i for i, q in enumerate(q_vals)}
    
    bid_depths = np.empty(n_paths, dtype=np.float64)
    ask_depths = np.empty(n_paths, dtype=np.float64)
    
    for i in range(n_paths):
        q = inventory[i]
        if q < -Q:
            q = -Q
        elif q > Q:
            q = Q
        q_idx = q_to_idx[q]
        
        db = policy_bid[n, q_idx]
        da = policy_ask[n, q_idx]
        
        # Boundary conditions
        if q >= Q:
            db = 0.0  # no bid at max inventory
        if q <= -Q:
            da = 0.0  # no ask at min inventory
        
        bid_depths[i] = db
        ask_depths[i] = da
    
    # Validate: ensure depths are >= 0 and not inf/nan
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, 1.0 / k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
