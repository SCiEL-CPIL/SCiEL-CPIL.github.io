"""Optimal market making with inventory risk: precomputed HJB solution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # 280s safety margin out of 300s

_cache = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal policy via backward induction (HJB discretization).
    
    State: (step n, inventory q)
    Action: choose bid_depth d_b >= 0 and ask_depth d_a >= 0
    Reward: earned depth from fills - (gamma/2)*sigma^2*q^2*dt
    
    The optimal depth on each side can be found by considering the trade-off
    between expected fill profit and inventory risk.
    
    For a given inventory q, the value function V(n, q) satisfies:
    V(n, q) = max_{d_b, d_a} [
        (1 - exp(-A*exp(-k*d_b)*dt)) * (d_b + V(n+1, q+1)) 
      + (1 - exp(-A*exp(-k*d_a)*dt)) * (-d_a + V(n+1, q-1))
      - (1 - exp(-A*exp(-k*d_b)*dt)) * (1 - exp(-A*exp(-k*d_a)*dt)) * (V(n+1, q))
      - (gamma/2) * sigma^2 * q^2 * dt
    ]
    Wait, that's not quite right. Let me reconsider.
    
    Actually, the reward structure is:
    - If bid is hit: earn d_b, inventory goes from q to q+1
    - If ask is hit: earn d_a (but this is a sale, so you're earning by selling high), inventory goes from q to q-1
    
    Wait, let me re-read the task: "A fill earns its depth and moves inventory one unit"
    
    So if bid is hit at depth d_b below mid, you buy at mid - d_b, so you earn d_b (relative to mid).
    If ask is hit at depth d_a above mid, you sell at mid + d_a, so you earn d_a.
    
    The objective is: sum of (depths earned) minus sum of (gamma/2)*sigma^2*q^2*dt
    
    So for a single step:
    - P(bid hit) = 1 - exp(-A*exp(-k*d_b)*dt)
    - P(ask hit) = 1 - exp(-A*exp(-k*d_a)*dt)
    
    Expected reward this step:
    P(bid hit) * d_b + P(ask hit) * d_a - (gamma/2)*sigma^2*q^2*dt
    
    Plus expected future value:
    P(bid hit) * P(no ask hit) * V(n+1, q+1) 
    + P(no bid hit) * P(ask hit) * V(n+1, q-1)
    + P(bid hit) * P(ask hit) * V(n+1, q)  # both hit, net inventory change is 0
    + P(no bid hit) * P(no ask hit) * V(n+1, q)
    
    Simplifying:
    = V(n+1, q) + P(bid hit)*P(no ask hit)*(V(n+1, q+1) - V(n+1, q)) 
                  + P(no bid hit)*P(ask hit)*(V(n+1, q-1) - V(n+1, q))
    
    So the HJB is:
    V(n, q) = max_{d_b, d_a} [
        P_b * d_b + P_a * d_a - (gamma/2)*sigma^2*q^2*dt
        + P_b*(1-P_a)*(V(n+1, q+1) - V(n+1, q))
        + (1-P_b)*P_a*(V(n+1, q-1) - V(n+1, q))
    ]
    
    where P_b = 1 - exp(-A*exp(-k*d_b)*dt), P_a = 1 - exp(-A*exp(-k*d_a)*dt)
    
    Since d_b and d_a appear separately in the objective (the cross terms don't couple them
    in a way that makes the joint optimization hard), we can optimize each independently.
    
    Let's verify: the terms involving d_b are:
    P_b * d_b + P_b*(1-P_a)*(V(n+1, q+1) - V(n+1, q))
    = P_b * [d_b + (1-P_a)*(V(n+1, q+1) - V(n+1, q))]
    
    And terms involving d_a:
    P_a * d_a + (1-P_b)*P_a*(V(n+1, q-1) - V(n+1, q))
    = P_a * [d_a + (1-P_b)*(V(n+1, q-1) - V(n+1, q))]
    
    Hmm, but P_b appears in the d_a term and vice versa, so they are coupled.
    
    However, we can use a grid search over depths for each side. The optimal depth
    is likely to be in a bounded range. Let's use a reasonable grid.
    
    Actually, for efficiency, let's note that the optimal d_b maximizes:
    f(d_b) = P_b * [d_b + (1-P_a)*delta_plus]
    where delta_plus = V(n+1, q+1) - V(n+1, q)
    
    And f'(d_b) = P_b' * [d_b + (1-P_a)*delta_plus] + P_b
    
    This is complex. Let's just use a grid search with a reasonable number of points.
    
    For each step and inventory, we'll search over a grid of depths for both sides.
    To keep it fast, we'll use a coarse grid and then refine if needed.
    """
    nq = 2 * Q + 1  # number of inventory states
    dt = T / steps
    
    # V[n, i] where i = q + Q, q in [-Q, Q]
    V = np.zeros((steps + 1, nq))
    bid_d = np.full((steps, nq), np.inf)
    ask_d = np.full((steps, nq), np.inf)
    
    # Terminal condition: V(steps, q) = 0 for all q
    # No terminal charge as per task
    
    # Grid of depths to search over
    # Optimal depth is typically O(1/k), so let's use a range
    max_depth = 5.0 / k  # reasonable upper bound
    num_depths = 50  # number of depth points
    
    depth_grid = np.linspace(0, max_depth, num_depths)
    
    for n in range(steps - 1, -1, -1):
        risk_term = (gamma / 2.0) * sigma**2 * dt
        for i in range(nq):
            q = i - Q
            
            # Boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
            if q == Q:
                # Only ask can be hit
                # Optimize d_a only
                best_val = -np.inf
                best_d_a = np.inf
                for d_a in depth_grid:
                    P_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                    val = P_a * d_a - risk_term * q**2
                    if i - 1 >= 0:  # q-1 >= -Q
                        val += (1.0 - P_a) * V[n + 1, i] + P_a * (V[n + 1, i - 1] - V[n + 1, i])
                    else:
                        # q-1 < -Q shouldn't happen if q >= -Q and q != Q
                        pass
                    if val > best_val:
                        best_val = val
                        best_d_a = d_a
                V[n, i] = best_val
                ask_d[n, i] = best_d_a
                bid_d[n, i] = np.inf
            elif q == -Q:
                # Only bid can be hit
                best_val = -np.inf
                best_d_b = np.inf
                for d_b in depth_grid:
                    P_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                    val = P_b * d_b - risk_term * q**2
                    if i + 1 < nq:  # q+1 <= Q
                        val += (1.0 - P_b) * V[n + 1, i] + P_b * (V[n + 1, i + 1] - V[n + 1, i])
                    else:
                        pass
                    if val > best_val:
                        best_val = val
                        best_d_b = d_b
                V[n, i] = best_val
                bid_d[n, i] = best_d_b
                ask_d[n, i] = np.inf
            else:
                # Both sides can be hit
                best_val = -np.inf
                best_d_b = np.inf
                best_d_a = np.inf
                delta_plus = V[n + 1, i + 1] - V[n + 1, i]
                delta_minus = V[n + 1, i - 1] - V[n + 1, i]
                
                for d_b in depth_grid:
                    P_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                    for d_a in depth_grid:
                        P_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                        
                        val = P_b * d_b + P_a * d_a - risk_term * q**2
                        val += P_b * (1.0 - P_a) * delta_plus
                        val += (1.0 - P_b) * P_a * delta_minus
                        
                        if val > best_val:
                            best_val = val
                            best_d_b = d_b
                            best_d_a = d_a
                
                V[n, i] = best_val
                bid_d[n, i] = best_d_b
                ask_d[n, i] = best_d_a
    
    return bid_d, ask_d


def _get_policy(params):
    """Get or compute the precomputed policy for a given parameter set."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _cache:
        if time.time() > _DEADLINE:
            # Return a safe default: quote at 1/k on both sides
            Q = params['Q']
            nq = 2 * Q + 1
            steps = params['steps']
            depth = np.full((steps, nq), 1.0 / params['k'])
            _cache[key] = (depth, depth.copy())
        else:
            _cache[key] = _solve_regime(
                params['sigma'], params['A'], params['k'],
                params['gamma'], params['T'], params['steps'], params['Q']
            )
    return _cache[key]


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) for all paths."""
    n_paths = len(inventory)
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    bid_table, ask_table = _get_policy(params)
    
    # Map inventory to index: i = q + Q
    inv_idx = inventory + Q
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # Ensure valid values: clip to [-3/k, inf]
    min_depth = -3.0 / params['k']
    bid_depths = np.where(bid_depths < min_depth, min_depth, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, min_depth, ask_depths)
    
    # Replace NaN with 1/k
    bid_depths = np.nan_to_num(bid_depths, nan=1.0 / params['k'], posinf=np.inf, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=1.0 / params['k'], posinf=np.inf, neginf=min_depth)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
