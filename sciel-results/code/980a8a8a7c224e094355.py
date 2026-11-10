"""Market making with inventory risk: policy iteration via backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import time

# Cache for solved policies per regime
_policy_cache = {}

def _solve_policy(params):
    """Solve the optimal policy via backward induction over inventory states."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _policy_cache:
        return _policy_cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # State: inventory q in [-Q, Q]
    # Value function V[n][q] = optimal expected objective from step n onwards
    # Policy: for each step n and inventory q, optimal (bid_depth, ask_depth)
    
    # For efficiency, we'll compute the optimal depth for each (n, q)
    # The fill probability for bid at depth d: p_bid = 1 - exp(-A * exp(-k*d) * dt)
    # For small dt: p_bid ≈ A * exp(-k*d) * dt
    
    # We'll do backward induction. At each step, given inventory q,
    # we choose bid_depth d_b and ask_depth d_a to maximize:
    # immediate reward + expected future value
    # 
    # The immediate reward from a fill:
    # - bid fill: earn d_b, inventory increases by 1
    # - ask fill: earn d_a, inventory decreases by 1
    # - no fill: earn 0
    #
    # Plus the running cost: -(gamma/2) * sigma^2 * q^2 * dt (inventory before fills)
    #
    # So the Bellman equation:
    # V[n][q] = -(gamma/2)*sigma^2*q^2*dt + max_{d_b, d_a} [
    #     (1-p_b)*(1-p_a)*V[n+1][q] + p_b*(1-p_a)*(d_b + V[n+1][q+1])
    #     + (1-p_b)*p_a*(d_a + V[n+1][q-1]) + p_b*p_a*(d_b + d_a + V[n+1][q])
    # ]
    # where p_b = 1-exp(-A*exp(-k*d_b)*dt), p_a = 1-exp(-A*exp(-k*d_a)*dt)
    # with boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    
    # For efficiency, note that the optimal d_b and d_a can be found by
    # 1D optimization since the problem decomposes somewhat.
    # But actually they're coupled through the cross term.
    
    # Let's use a grid search over depths for each (n, q).
    # Depth range: [0.01, 5.0/k] should cover reasonable values.
    
    # Number of depth grid points
    n_depth = 50
    max_depth = 5.0 / k
    min_depth = 0.01
    depths_grid = np.linspace(min_depth, max_depth, n_depth)
    
    # V[n][q] where n is step index (0 to steps), q is inventory (-Q to Q)
    # We'll store as 2D array: V[n, q+Q]
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Policy: store optimal depths for each (n, q)
    # pol_bid[n, q+Q], pol_ask[n, q+Q]
    pol_bid = np.zeros((steps, 2 * Q + 1))
    pol_ask = np.zeros((steps, 2 * Q + 1))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for qi in range(2 * Q + 1):
            q = qi - Q
            # Running cost
            cost = -(gamma / 2.0) * sigma * sigma * q * q * dt
            
            if n == steps - 1:
                # Last step: V[steps][q] = 0
                V_next = np.zeros(2 * Q + 1)
            else:
                V_next = V[n + 1]
            
            best_val = -np.inf
            best_db = min_depth
            best_da = min_depth
            
            # Grid search over (d_b, d_a)
            # To speed up, we can first optimize d_b given d_a and vice versa,
            # but let's do a coarse grid first then refine.
            
            for db in depths_grid:
                p_b = 1.0 - np.exp(-A * np.exp(-k * db) * dt)
                # At q = Q, no bid fill possible
                if q == Q:
                    p_b = 0.0
                
                for da in depths_grid:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * da) * dt)
                    # At q = -Q, no ask fill possible
                    if q == -Q:
                        p_a = 0.0
                    
                    # Compute expected value
                    # No fill: (1-p_b)(1-p_a) * V_next[q]
                    # Bid only: p_b(1-p_a) * (db + V_next[q+1]) if q < Q
                    # Ask only: (1-p_b)p_a * (da + V_next[q-1]) if q > -Q
                    # Both: p_b*p_a * (db + da + V_next[q])
                    
                    val = (1.0 - p_b) * (1.0 - p_a) * V_next[qi]
                    
                    if q < Q and p_b > 0:
                        val += p_b * (1.0 - p_a) * (db + V_next[qi + 1])
                    
                    if q > -Q and p_a > 0:
                        val += (1.0 - p_b) * p_a * (da + V_next[qi - 1])
                    
                    if q < Q and q > -Q and p_b > 0 and p_a > 0:
                        val += p_b * p_a * (db + da + V_next[qi])
                    
                    total = cost + val
                    
                    if total > best_val:
                        best_val = total
                        best_db = db
                        best_da = da
            
            V[n, qi] = best_val
            pol_bid[n, qi] = best_db
            pol_ask[n, qi] = best_da
    
    _policy_cache[key] = (V, pol_bid, pol_ask)
    return (V, pol_bid, pol_ask)


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int32)
    
    # Get or compute the policy
    try:
        V, pol_bid, pol_ask = _solve_policy(params)
    except Exception:
        # Fallback: simple heuristic
        base_depth = 1.0 / k
        # Skew based on inventory
        skew = gamma * sigma * sigma * inv.astype(np.float64) * (T - t) / (k * A)
        bid_depths = np.clip(base_depth - skew, 0.01, 5.0 / k)
        ask_depths = np.clip(base_depth + skew, 0.01, 5.0 / k)
        # Boundary conditions
        bid_depths[inv >= Q] = np.inf
        ask_depths[inv <= -Q] = np.inf
        return bid_depths, ask_depths
    
    # Determine current step index
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = min(max(step_idx, 0), steps - 1)
    
    # Get depths from policy
    bid_depths = np.empty(n_paths, dtype=np.float64)
    ask_depths = np.empty(n_paths, dtype=np.float64)
    
    for i in range(n_paths):
        q = inv[i]
        qi = q + Q
        # Clamp to valid range
        qi = min(max(qi, 0), 2 * Q)
        
        bd = pol_bid[step_idx, qi]
        ad = pol_ask[step_idx, qi]
        
        # Boundary conditions
        if q >= Q:
            bd = np.inf
        if q <= -Q:
            ad = np.inf
        
        bid_depths[i] = bd
        ask_depths[i] = ad
    
    # Ensure valid depths (above -3/k, not -inf, not NaN)
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, np.inf)
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
