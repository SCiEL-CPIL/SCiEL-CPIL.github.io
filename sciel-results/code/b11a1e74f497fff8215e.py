import numpy as np
from functools import lru_cache
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill

_dp_cache = {}
_heuristic_fallback = False

def _heuristic_quote(t, inventory, mid, params):
    """Fast heuristic fallback if DP doesn't finish in time."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    n = int(round(t / dt))
    remaining = (steps - n) * dt
    
    # Base depth from log-optimal form: depth ~ (1/k) * log(A * dt / something)
    # Calibrate to 1/k as base, adjust for inventory skew and time
    base_depth = 1.0 / k
    
    # Inventory skew: quote tighter on the side that reduces inventory
    # If inventory > 0, we want to sell (ask) more aggressively -> tighter ask
    # If inventory < 0, we want to buy (bid) more aggressively -> tighter bid
    skew = np.clip(inventory.astype(np.float64) / Q, -1.0, 1.0)
    
    # Time decay: tighter quotes early, wider later (less urgency)
    time_factor = np.sqrt(remaining / T)
    
    # Risk aversion: higher gamma -> wider quotes to reduce inventory risk
    risk_factor = 1.0 + gamma * sigma * sigma * 0.1
    
    bid_depths = base_depth * risk_factor * time_factor * (1.0 - 0.3 * skew)
    ask_depths = base_depth * risk_factor * time_factor * (1.0 + 0.3 * skew)
    
    # Ensure minimum depth
    min_depth = 1e-4
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Boundary conditions
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths

def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """Precompute optimal bid/ask depths for each (time_step, inventory) pair."""
    global _heuristic_fallback
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # V[t, q] = optimal expected value from step t with inventory q
    # q_idx = q + Q, so q = q_idx - Q
    V = np.zeros((n_steps + 1, n_inv), dtype=np.float64)
    bid_depths = np.zeros((n_steps, n_inv), dtype=np.float64)
    ask_depths = np.zeros((n_steps, n_inv), dtype=np.float64)
    
    # Terminal: V[T, q] = 0 (no terminal charge)
    
    # Grid of candidate depths to evaluate - use fewer candidates for speed
    max_depth = 5.0 / k
    n_candidates = 80
    candidate_depths = np.linspace(1e-4, max_depth, n_candidates)
    
    # Precompute fill probabilities for each candidate depth
    # p_fill[d] = 1 - exp(-A * exp(-k * d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * candidate_depths) * dt)
    
    # Work backwards in time
    for t in range(n_steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            _heuristic_fallback = True
            return None, None, dt
        
        # For each inventory level
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory risk penalty for this step
            risk_penalty = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # If at boundary, one side has no fill
            if q == Q:
                # No bid fill possible
                bid_depths[t, q_idx] = np.inf
                # Find optimal ask depth
                # val = a * p_ask - risk_penalty + V[t+1, q-1]
                # Maximize over a
                vals = candidate_depths * fill_probs - risk_penalty + V[t + 1, q_idx - 1]
                best_idx = np.argmax(vals)
                ask_depths[t, q_idx] = candidate_depths[best_idx]
                V[t, q_idx] = vals[best_idx]
            elif q == -Q:
                # No ask fill possible
                ask_depths[t, q_idx] = np.inf
                # Find optimal bid depth
                # val = b * p_bid - risk_penalty + V[t+1, q+1]
                vals = candidate_depths * fill_probs - risk_penalty + V[t + 1, q_idx + 1]
                best_idx = np.argmax(vals)
                bid_depths[t, q_idx] = candidate_depths[best_idx]
                V[t, q_idx] = vals[best_idx]
            else:
                # Both sides can fill
                # For each bid candidate b_i and ask candidate a_j:
                # p_b = fill_probs[i], p_a = fill_probs[j]
                # p_bid_only = p_b * (1 - p_a)
                # p_ask_only = p_a * (1 - p_b)  
                # p_both = p_b * p_a
                # p_none = (1 - p_b) * (1 - p_a)
                # val = b_i * p_b + a_j * p_a - risk_penalty 
                #      + p_bid_only * V[t+1, q+1]
                #      + p_ask_only * V[t+1, q-1]
                #      + (p_both + p_none) * V[t+1, q]
                
                V_next_q_plus = V[t + 1, q_idx + 1]
                V_next_q_minus = V[t + 1, q_idx - 1]
                V_next_q = V[t + 1, q_idx]
                
                # Vectorize over bid candidates for each ask candidate
                best_val = -np.inf
                best_bid = candidate_depths[0]
                best_ask = candidate_depths[0]
                
                for j in range(n_candidates):
                    a = candidate_depths[j]
                    p_a = fill_probs[j]
                    one_minus_p_a = 1.0 - p_a
                    
                    # For all bid candidates at once
                    p_b = fill_probs  # (n_candidates,)
                    one_minus_p_b = 1.0 - p_b
                    
                    p_bid_only = p_b * one_minus_p_a
                    p_ask_only = p_a * one_minus_p_b
                    p_both = p_b * p_a
                    p_none = one_minus_p_b * one_minus_p_a
                    
                    vals = candidate_depths * p_b + a * p_a - risk_penalty
                    vals += p_bid_only * V_next_q_plus
                    vals += p_ask_only * V_next_q_minus
                    vals += (p_both + p_none) * V_next_q
                    
                    best_j = np.argmax(vals)
                    if vals[best_j] > best_val:
                        best_val = vals[best_j]
                        best_bid = candidate_depths[best_j]
                        best_ask = a
                
                bid_depths[t, q_idx] = best_bid
                ask_depths[t, q_idx] = best_ask
                V[t, q_idx] = best_val
    
    return bid_depths, ask_depths, dt


def _get_dp_tables(params):
    """Get or compute DP tables for given params."""
    global _heuristic_fallback
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _dp_cache:
        sigma, A, k, gamma, T, steps, Q = key
        bid_depths, ask_depths, dt = _compute_dp_table(sigma, A, k, gamma, T, steps, Q)
        if bid_depths is None:
            _heuristic_fallback = True
            _dp_cache[key] = (None, None, dt)
        else:
            _dp_cache[key] = (bid_depths, ask_depths, dt)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy from dynamic programming."""
    global _heuristic_fallback
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Check deadline
    if time.time() > _DEADLINE:
        return _heuristic_quote(t, inventory, mid, params)
    
    dt = T / steps
    n = int(round(t / dt))
    n = min(n, steps - 1)  # clamp to valid step index
    
    # Get precomputed tables
    bid_table, ask_table, _ = _get_dp_tables(params)
    
    # If DP failed, use heuristic
    if bid_table is None or _heuristic_fallback:
        return _heuristic_quote(t, inventory, mid, params)
    
    # Map inventory to index: q_idx = q + Q
    q_idx = inventory.astype(np.int64) + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Look up optimal depths using advanced indexing
    bid_depths = bid_table[n, q_idx].astype(np.float64)
    ask_depths = ask_table[n, q_idx].astype(np.float64)
    
    # Ensure valid values
    min_depth = 1e-4
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_depth), min_depth, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_depth), min_depth, ask_depths)
    
    # Handle boundary conditions: no bid at Q, no ask at -Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
