import numpy as np
from functools import lru_cache
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill

_dp_cache = {}

def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """Precompute optimal bid/ask depths for each (time_step, inventory) pair."""
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # V[t, q] = optimal expected value from step t with inventory q
    # q_idx = q + Q, so q = q_idx - Q
    V = np.zeros((n_steps + 1, n_inv), dtype=np.float64)
    bid_depths = np.zeros((n_steps, n_inv), dtype=np.float64)
    ask_depths = np.zeros((n_steps, n_inv), dtype=np.float64)
    
    # Terminal: V[T, q] = 0 (no terminal charge)
    
    # Grid of candidate depths to evaluate
    # We'll search over a reasonable range of depths
    max_depth = 5.0 / k  # depths beyond this give negligible fill prob
    n_candidates = 200
    candidate_depths = np.linspace(1e-6, max_depth, n_candidates)
    
    # Work backwards in time
    for t in range(n_steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: use heuristic
            break
        remaining = n_steps - t
        # For each inventory level
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory risk penalty for this step
            risk_penalty = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # If at boundary, one side has no fill
            if q == Q:
                # No bid fill possible
                best_val = -risk_penalty + V[t + 1, q_idx]
                bid_depths[t, q_idx] = np.inf
                # Find optimal ask depth
                best_ask = candidate_depths[0]
                for a in candidate_depths:
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    # After ask fill, inventory decreases by 1
                    next_q = q - 1
                    next_q_idx = next_q + Q
                    val = a * p_ask - risk_penalty + V[t + 1, next_q_idx]
                    if val > best_val:
                        best_val = val
                        best_ask = a
                ask_depths[t, q_idx] = best_ask
                V[t, q_idx] = best_val
            elif q == -Q:
                # No ask fill possible
                best_val = -risk_penalty + V[t + 1, q_idx]
                ask_depths[t, q_idx] = np.inf
                # Find optimal bid depth
                best_bid = candidate_depths[0]
                for b in candidate_depths:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    # After bid fill, inventory increases by 1
                    next_q = q + 1
                    next_q_idx = next_q + Q
                    val = b * p_bid - risk_penalty + V[t + 1, next_q_idx]
                    if val > best_val:
                        best_val = val
                        best_bid = b
                bid_depths[t, q_idx] = best_bid
                V[t, q_idx] = best_val
            else:
                # Both sides can fill
                best_val = -np.inf
                best_bid = candidate_depths[0]
                best_ask = candidate_depths[0]
                for b in candidate_depths:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    for a in candidate_depths:
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                        # Expected next inventory: q + p_bid - p_ask
                        # But we need to handle the discrete nature
                        # With small dt, probability of both fills is very small
                        # Approximate: expected change = p_bid - p_ask
                        # For exact: P(both) = p_bid * p_ask
                        p_both = p_bid * p_ask
                        p_bid_only = p_bid * (1 - p_ask)
                        p_ask_only = p_ask * (1 - p_bid)
                        p_none = (1 - p_bid) * (1 - p_ask)
                        
                        # Next inventory states
                        val = b * p_bid + a * p_ask - risk_penalty
                        # Continuation value (approximate with expected)
                        # For small dt, we can use linear approximation
                        # E[V[t+1, q']] where q' = q + 1 (bid only), q-1 (ask only), q+1-1=q (both), q (none)
                        val += p_bid_only * V[t + 1, q_idx + 1]
                        val += p_ask_only * V[t + 1, q_idx - 1]
                        val += p_both * V[t + 1, q_idx]  # q + 1 - 1 = q
                        val += p_none * V[t + 1, q_idx]
                        
                        if val > best_val:
                            best_val = val
                            best_bid = b
                            best_ask = a
                bid_depths[t, q_idx] = best_bid
                ask_depths[t, q_idx] = best_ask
                V[t, q_idx] = best_val
    
    return bid_depths, ask_depths, dt


def _get_dp_tables(params):
    """Get or compute DP tables for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _dp_cache:
        sigma, A, k, gamma, T, steps, Q = key
        bid_depths, ask_depths, dt = _compute_dp_table(sigma, A, k, gamma, T, steps, Q)
        _dp_cache[key] = (bid_depths, ask_depths, dt)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy from dynamic programming."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n = int(round(t / dt))
    n = min(n, steps - 1)  # clamp to valid step index
    
    # Get precomputed tables
    bid_table, ask_table, _ = _get_dp_tables(params)
    
    # Map inventory to index: q_idx = q + Q
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Look up optimal depths
    bid_depths = bid_table[n, q_idx].astype(np.float64)
    ask_depths = ask_table[n, q_idx].astype(np.float64)
    
    # Ensure valid values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths <= 0), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths <= 0), 1.0/k, ask_depths)
    
    # Handle boundary conditions: no bid at Q, no ask at -Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
