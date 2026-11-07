"""Optimal market making via HJB backward induction with pre-computed lookup tables."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}


def _solve_optimal_depth(C, alpha, k):
    """
    Finds optimal depth x >= 0 maximizing (1 - exp(-alpha * exp(-k*x))) * (x + C)
    C is the value difference V(q+1) - V(q) for bid, or V(q-1) - V(q) for ask.
    """
    if alpha <= 0:
        return 0.0
    
    def neg_obj(x):
        if x < 0:
            return 1e10
        p = 1.0 - np.exp(-alpha * np.exp(-k * x))
        return -p * (x + C)
    
    # Initial guess based on C
    if C > 0:
        # We want fills, so moderate depth
        x0 = 1.0 / k
    else:
        # We want to avoid fills, so deep
        x0 = 2.0 / k
    
    # Bounds: depth should be reasonable
    x_min = 0.0
    x_max = 5.0 / k
    
    res = minimize_scalar(neg_obj, bounds=(x_min, x_max), method='bounded',
                          options={'xatol': 1e-8})
    x_opt = res.x
    if x_opt < 0:
        x_opt = 0.0
    return float(x_opt)


def _compute_policy(params):
    """
    Pre-compute the optimal policy table via backward induction.
    Returns a dict with keys (step, inventory) -> (bid_depth, ask_depth)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # State: inventory in [-Q, Q]
    # Value function V[n, q] for n = 0..steps
    # V[steps, q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Policy tables: bid_depth[n, q], ask_depth[n, q]
    bid_depth = np.zeros((steps, 2 * Q + 1))
    ask_depth = np.zeros((steps, 2 * Q + 1))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            q_idx = q + Q  # index into V array
            
            # Inventory cost for this step
            cost = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # Try bid side
            # If we quote bid at depth d, with prob p_bid we get a fill
            # Fill earns d, inventory becomes q+1 (if q < Q)
            # If no fill, inventory stays q
            # Value if bid fill: d + V[n+1, q+1] (if q < Q)
            # Value if no bid fill: V[n+1, q]
            
            # Similarly for ask side
            # Ask fill earns d, inventory becomes q-1 (if q > -Q)
            
            # We need to optimize bid_depth and ask_depth jointly
            # But they are independent given the value function
            # Actually, the fill events are independent, so we can separate
            
            # Expected value from bid side (ignoring ask for now):
            # E_bid = (1 - exp(-A * exp(-k * d_b))) * (d_b + V[n+1, q+1] - V[n+1, q]) + V[n+1, q]
            #       = V[n+1, q] + (1 - exp(-A * exp(-k * d_b))) * (d_b + V[n+1, q+1] - V[n+1, q])
            
            # Similarly for ask:
            # E_ask = V[n+1, q] + (1 - exp(-A * exp(-k * d_a))) * (d_a + V[n+1, q-1] - V[n+1, q])
            
            # Total expected value:
            # E = -cost + E_bid + E_ask - V[n+1, q]
            #   = -cost + (1 - exp(-A * exp(-k * d_b))) * (d_b + V[n+1, q+1] - V[n+1, q])
            #           + (1 - exp(-A * exp(-k * d_a))) * (d_a + V[n+1, q-1] - V[n+1, q])
            
            # Wait, this isn't quite right because the fills are independent.
            # Let me reconsider.
            
            # The mid moves, but for the depth optimization at this step,
            # we condition on the mid not moving (or the depth is relative to mid).
            # The value function V[n+1, q'] already accounts for future mid moves.
            
            # Let's define:
            # C_bid = V[n+1, q+1] - V[n+1, q] if q < Q, else -inf (no bid fill at q=Q)
            # C_ask = V[n+1, q-1] - V[n+1, q] if q > -Q, else -inf (no ask fill at q=-Q)
            
            # If q == Q: no bid fill possible, so bid_depth = inf (or 0, doesn't matter)
            # If q == -Q: no ask fill possible
            
            if q == Q:
                bid_depth[n, q_idx] = np.inf
                # Optimize ask only
                C_ask = V[n+1, q_idx - 1] - V[n+1, q_idx]
                d_a = _solve_optimal_depth(C_ask, A, k)
                ask_depth[n, q_idx] = d_a
                V[n, q_idx] = -cost + (1 - np.exp(-A * np.exp(-k * d_a))) * (d_a + C_ask) + V[n+1, q_idx]
            elif q == -Q:
                ask_depth[n, q_idx] = np.inf
                C_bid = V[n+1, q_idx + 1] - V[n+1, q_idx]
                d_b = _solve_optimal_depth(C_bid, A, k)
                bid_depth[n, q_idx] = d_b
                V[n, q_idx] = -cost + (1 - np.exp(-A * np.exp(-k * d_b))) * (d_b + C_bid) + V[n+1, q_idx]
            else:
                C_bid = V[n+1, q_idx + 1] - V[n+1, q_idx]
                C_ask = V[n+1, q_idx - 1] - V[n+1, q_idx]
                
                d_b = _solve_optimal_depth(C_bid, A, k)
                d_a = _solve_optimal_depth(C_ask, A, k)
                
                bid_depth[n, q_idx] = d_b
                ask_depth[n, q_idx] = d_a
                
                V[n, q_idx] = -cost + V[n+1, q_idx] \
                    + (1 - np.exp(-A * np.exp(-k * d_b))) * (d_b + C_bid) \
                    + (1 - np.exp(-A * np.exp(-k * d_a))) * (d_a + C_ask)
    
    return bid_depth, ask_depth


def _get_optimal_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback: use simple heuristic
            depth = 1.0 / params['k']
            Q = params['Q']
            steps = params['steps']
            bid = np.full((steps, 2 * Q + 1), depth)
            ask = np.full((steps, 2 * Q + 1), depth)
            _POLICY_CACHE[key] = (bid, ask)
        else:
            bid, ask = _compute_policy(params)
            _POLICY_CACHE[key] = (bid, ask)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths for each path.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()
    
    bid_table, ask_table = _get_optimal_table(params)
    
    steps = params['steps']
    Q = params['Q']
    n = int(round(t / (params['T'] / steps)))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to index
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]
    
    # Ensure valid values
    bid_depths = np.where(bid_depths < 0, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.0, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
