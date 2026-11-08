"""Optimal market making with inventory risk using pre-computed backward-induction policy."""
import numpy as np
import time

# Module-level cache for policy tables
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIME_BUDGET = 310  # seconds, safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth for each (step, inventory) via backward induction.
    Uses vectorized grid search + local refinement for speed.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: store actual float depths
    bid_policy = np.zeros((steps, n_inv), dtype=np.float64)
    ask_policy = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Depth bounds for optimization
    # Max depth where prob is negligible
    max_depth = 15.0 / k
    
    # Precompute grid for coarse search
    n_coarse = 200
    depth_grid = np.linspace(0, max_depth, n_coarse)
    
    # Precompute fill probabilities on the grid
    exp_term = np.exp(-k * depth_grid)
    prob_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Precompute q values
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            break
        
        # Inventory holding cost for this step
        hold_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            V_next_q = V[n+1, q_idx]
            
            if q < Q:
                V_next_q_plus = V[n+1, q_idx + 1]
            else:
                V_next_q_plus = 0.0  # won't be used
            
            if q > -Q:
                V_next_q_minus = V[n+1, q_idx - 1]
            else:
                V_next_q_minus = 0.0  # won't be used
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_bid_depth = 0.0
            best_ask_depth = 0.0
            best_val = -np.inf
            
            if can_bid and can_ask:
                # Both sides active
                # Vectorized grid search
                PB = prob_grid[:, None]  # (n_coarse, 1)
                PA = prob_grid[None, :]  # (1, n_coarse)
                B = depth_grid[:, None]  # (n_coarse, 1)
                A_ = depth_grid[None, :]  # (1, n_coarse)
                
                obj_grid = ((1-PB)*(1-PA)*V_next_q 
                           + PB*(1-PA)*(B + V_next_q_plus)
                           + (1-PB)*PA*(A_ + V_next_q_minus)
                           + PB*PA*(B + A_ + V_next_q)
                           - hold_cost[q_idx])
                
                # Find best grid point
                flat_idx = np.argmax(obj_grid)
                b0 = depth_grid[flat_idx // n_coarse]
                a0 = depth_grid[flat_idx % n_coarse]
                
                # Local refinement using gradient-based approach
                # We'll do a few iterations of alternating optimization
                b_curr = b0
                a_curr = a0
                
                for _iter in range(5):
                    # Optimize b with a fixed
                    # Use a small grid around current b
                    b_range = np.linspace(max(0, b_curr - 2.0/k), min(max_depth, b_curr + 2.0/k), 100)
                    p_b_range = 1.0 - np.exp(-A * np.exp(-k * b_range) * dt)
                    p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a_curr) * dt)
                    
                    obj_b = ((1-p_b_range)*(1-p_a_curr)*V_next_q 
                            + p_b_range*(1-p_a_curr)*(b_range + V_next_q_plus)
                            + (1-p_b_range)*p_a_curr*(a_curr + V_next_q_minus)
                            + p_b_range*p_a_curr*(b_range + a_curr + V_next_q)
                            - hold_cost[q_idx])
                    
                    b_curr = b_range[np.argmax(obj_b)]
                    
                    # Optimize a with b fixed
                    a_range = np.linspace(max(0, a_curr - 2.0/k), min(max_depth, a_curr + 2.0/k), 100)
                    p_a_range = 1.0 - np.exp(-A * np.exp(-k * a_range) * dt)
                    p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b_curr) * dt)
                    
                    obj_a = ((1-p_b_curr)*(1-p_a_range)*V_next_q 
                            + p_b_curr*(1-p_a_range)*(b_curr + V_next_q_plus)
                            + (1-p_b_curr)*p_a_range*(a_range + V_next_q_minus)
                            + p_b_curr*p_a_range*(b_curr + a_range + V_next_q)
                            - hold_cost[q_idx])
                    
                    a_curr = a_range[np.argmax(obj_a)]
                
                best_bid_depth = b_curr
                best_ask_depth = a_curr
                
                # Recompute best_val
                p_b = 1.0 - np.exp(-A * np.exp(-k * best_bid_depth) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * best_ask_depth) * dt)
                best_val = ((1-p_b)*(1-p_a)*V_next_q 
                           + p_b*(1-p_a)*(best_bid_depth + V_next_q_plus)
                           + (1-p_b)*p_a*(best_ask_depth + V_next_q_minus)
                           + p_b*p_a*(best_bid_depth + best_ask_depth + V_next_q)
                           - hold_cost[q_idx])
                
            elif can_bid:
                # Only bid side active
                # Coarse grid
                obj_b = ((1-prob_grid)*V_next_q 
                        + prob_grid*(depth_grid + V_next_q_plus)
                        - hold_cost[q_idx])
                b_idx = np.argmax(obj_b)
                b_curr = depth_grid[b_idx]
                
                # Local refinement
                b_range = np.linspace(max(0, b_curr - 2.0/k), min(max_depth, b_curr + 2.0/k), 100)
                p_b_range = 1.0 - np.exp(-A * np.exp(-k * b_range) * dt)
                obj_b_refine = ((1-p_b_range)*V_next_q 
                               + p_b_range*(b_range + V_next_q_plus)
                               - hold_cost[q_idx])
                b_curr = b_range[np.argmax(obj_b_refine)]
                
                best_bid_depth = b_curr
                best_ask_depth = 0.0
                p_b = 1.0 - np.exp(-A * np.exp(-k * best_bid_depth) * dt)
                best_val = (1-p_b)*V_next_q + p_b*(best_bid_depth + V_next_q_plus) - hold_cost[q_idx]
                
            elif can_ask:
                # Only ask side active
                # Coarse grid
                obj_a = ((1-prob_grid)*V_next_q 
                        + prob_grid*(depth_grid + V_next_q_minus)
                        - hold_cost[q_idx])
                a_idx = np.argmax(obj_a)
                a_curr = depth_grid[a_idx]
                
                # Local refinement
                a_range = np.linspace(max(0, a_curr - 2.0/k), min(max_depth, a_curr + 2.0/k), 100)
                p_a_range = 1.0 - np.exp(-A * np.exp(-k * a_range) * dt)
                obj_a_refine = ((1-p_a_range)*V_next_q 
                               + p_a_range*(a_range + V_next_q_minus)
                               - hold_cost[q_idx])
                a_curr = a_range[np.argmax(obj_a_refine)]
                
                best_ask_depth = a_curr
                best_bid_depth = 0.0
                p_a = 1.0 - np.exp(-A * np.exp(-k * best_ask_depth) * dt)
                best_val = (1-p_a)*V_next_q + p_a*(best_ask_depth + V_next_q_minus) - hold_cost[q_idx]
                
            else:
                best_bid_depth = 0.0
                best_ask_depth = 0.0
                best_val = -hold_cost[q_idx] + V_next_q
            
            V[n, q_idx] = best_val
            bid_policy[n, q_idx] = best_bid_depth
            ask_policy[n, q_idx] = best_ask_depth
    
    return bid_policy, ask_policy


def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each element is the distance below/above mid to quote.
    np.inf means no quote.
    """
    # Determine step index from t
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(max(n, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index
    inv_idx = inventory + Q  # inventory in [-Q, Q], so inv_idx in [0, 2Q]
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # At q=Q, no bid fill possible -> set bid to inf
    # At q=-Q, no ask fill possible -> set ask to inf
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure values are valid: >= 0, not NaN, not -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    # Ensure shape is (4000,)
    if len(bid_depths) != 4000:
        # Pad or truncate
        if len(bid_depths) < 4000:
            pad = np.full(4000 - len(bid_depths), np.inf)
            bid_depths = np.concatenate([bid_depths, pad])
            ask_depths = np.concatenate([ask_depths, pad])
        else:
            bid_depths = bid_depths[:4000]
            ask_depths = ask_depths[:4000]
    
    return bid_depths, ask_depths
