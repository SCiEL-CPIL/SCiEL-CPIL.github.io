# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal policy via backward induction on a coarse depth grid.
    Returns a dict mapping (step, inventory) to (bid_depth, ask_depth).
    """
    dt = T / steps
    # Coarse grid of depths
    # Max depth of interest: if depth > 5/k, prob is negligible
    # Min depth: 0
    # Use 200 points
    n_depths = 200
    depths = np.linspace(0.0, 5.0 / k, n_depths)
    
    # Precompute fill probabilities for all depths
    # p_fill[depth_idx] = 1 - exp(-A * exp(-k * depth))
    exp_term = np.exp(-k * depths)
    p_fill = 1.0 - np.exp(-A * exp_term)
    
    # Value function: V[step, q] where q in [-Q, Q]
    # q_index = q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    Policy = {}
    
    # Terminal condition: V[steps, q] = 0 for all q
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            V_next_q = V[n + 1, q_idx]
            
            # Inventory cost
            cost = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            best_val = -np.inf
            best_bid = 0.0
            best_ask = 0.0
            
            # Try all bid depths
            for bid_idx, bid_d in enumerate(depths):
                if q >= Q:
                    # No bid fill possible
                    p_bid = 0.0
                else:
                    p_bid = p_fill[bid_idx]
                
                # Try all ask depths
                for ask_idx, ask_d in enumerate(depths):
                    if q <= -Q:
                        # No ask fill possible
                        p_ask = 0.0
                    else:
                        p_ask = p_fill[ask_idx]
                    
                    # Expected value
                    # Outcomes:
                    # 1. No fill: prob (1-p_bid)*(1-p_ask), inventory q, reward 0
                    # 2. Bid fill: prob p_bid*(1-p_ask), inventory q+1, reward bid_d
                    # 3. Ask fill: prob (1-p_bid)*p_ask, inventory q-1, reward ask_d
                    # 4. Both fill: prob p_bid*p_ask, inventory q, reward bid_d + ask_d
                    
                    v_no = V[n + 1, q_idx]
                    v_bid = V[n + 1, q_idx + 1] if q < Q else -np.inf
                    v_ask = V[n + 1, q_idx - 1] if q > -Q else -np.inf
                    v_both = V[n + 1, q_idx]
                    
                    ev = 0.0
                    ev += (1.0 - p_bid) * (1.0 - p_ask) * v_no
                    if q < Q:
                        ev += p_bid * (1.0 - p_ask) * (bid_d + v_bid)
                    if q > -Q:
                        ev += (1.0 - p_bid) * p_ask * (ask_d + v_ask)
                    if q < Q and q > -Q:
                        ev += p_bid * p_ask * (bid_d + ask_d + v_both)
                    
                    total = ev - cost
                    
                    if total > best_val:
                        best_val = total
                        best_bid = bid_d
                        best_ask = ask_d
            
            V[n, q_idx] = best_val
            Policy[(n, q)] = (best_bid, best_ask)
            
            # Check deadline
            if time.time() > _DEADLINE - 10.0:
                # Fallback to simple heuristic
                return None
    
    return Policy


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    
    policy = _POLICY_CACHE[key]
    
    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)
    
    bid_depths = np.empty(len(inventory))
    ask_depths = np.empty(len(inventory))
    
    if policy is None:
        # Fallback heuristic
        for i, q in enumerate(inventory):
            # Simple heuristic: shift depth based on inventory
            shift = -gamma * sigma * sigma * q / (2.0 * A * k)
            base = 1.0 / k
            bid_d = max(0.0, base + shift)
            ask_d = max(0.0, base - shift)
            bid_depths[i] = bid_d
            ask_depths[i] = ask_d
    else:
        for i, q in enumerate(inventory):
            q_int = int(np.clip(q, -Q, Q))
            bid_d, ask_d = policy.get((n, q_int), (1.0 / k, 1.0 / k))
            bid_depths[i] = bid_d
            ask_depths[i] = ask_d
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
