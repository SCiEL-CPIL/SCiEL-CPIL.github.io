"""Optimal market making with inventory risk using pre-computed backward-induction policy."""
import numpy as np
import time
from scipy.optimize import minimize_scalar

# Module-level cache for policy tables
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIME_BUDGET = 320  # seconds, safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth for each (step, inventory) via backward induction.
    Uses continuous depth optimization via scipy minimize_scalar for higher precision.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: store actual float depths
    bid_policy = np.zeros((steps, n_inv), dtype=np.float64)
    ask_policy = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Depth bounds for optimization
    max_depth = 15.0 / k
    
    # Precompute for speed: we'll use a helper to evaluate the objective at a given depth
    def fill_prob(depth):
        """Fill probability for a given depth."""
        if depth < 0:
            depth = 0.0
        return 1.0 - np.exp(-A * np.exp(-k * depth) * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            break
        
        # Inventory holding cost for this step
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
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
                # Objective: maximize over (b, a) >= 0
                # f(b, a) = (1-p_b)(1-p_a)*V_next_q 
                #          + p_b(1-p_a)*(b + V_next_q_plus)
                #          + (1-p_b)*p_a*(a + V_next_q_minus)
                #          + p_b*p_a*(b + a + V_next_q)
                #          - hold_cost[q_idx]
                
                # Since the problem separates in a clever way, let's solve jointly
                # but we can also note that for fixed a, the optimal b can be found, and vice versa.
                # However, due to coupling through p_b*p_a term, it's not fully separable.
                # Let's do a 2D optimization with a grid + local refinement approach.
                
                # First, do a coarse grid search to find approximate optimum
                n_coarse = 50
                b_grid = np.linspace(0, max_depth, n_coarse)
                a_grid = np.linspace(0, max_depth, n_coarse)
                
                p_b_grid = 1.0 - np.exp(-A * np.exp(-k * b_grid) * dt)
                p_a_grid = 1.0 - np.exp(-A * np.exp(-k * a_grid) * dt)
                
                # Compute objective on grid
                PB = p_b_grid[:, None]
                PA = p_a_grid[None, :]
                B = b_grid[:, None]
                A_ = a_grid[None, :]
                
                obj_grid = ((1-PB)*(1-PA)*V_next_q 
                           + PB*(1-PA)*(B + V_next_q_plus)
                           + (1-PB)*PA*(A_ + V_next_q_minus)
                           + PB*PA*(B + A_ + V_next_q)
                           - hold_cost[q_idx])
                
                # Find best grid point
                flat_idx = np.argmax(obj_grid)
                b0 = b_grid[flat_idx // n_coarse]
                a0 = a_grid[flat_idx % n_coarse]
                
                # Now do local refinement using scipy
                # Optimize over b with a fixed at a0, then over a with b fixed, iterate
                b_curr = b0
                a_curr = a0
                
                for _iter in range(3):
                    # Optimize b with a fixed
                    def neg_obj_b(b):
                        p_b = 1.0 - np.exp(-A * np.exp(-k * max(0, b)) * dt)
                        p_a = 1.0 - np.exp(-A * np.exp(-k * max(0, a_curr)) * dt)
                        val = ((1-p_b)*(1-p_a)*V_next_q 
                               + p_b*(1-p_a)*(max(0,b) + V_next_q_plus)
                               + (1-p_b)*p_a*(a_curr + V_next_q_minus)
                               + p_b*p_a*(max(0,b) + a_curr + V_next_q)
                               - hold_cost[q_idx])
                        return -val
                    
                    res_b = minimize_scalar(neg_obj_b, bounds=(0, max_depth), method='bounded', options={'xatol': 1e-6})
                    b_curr = max(0, res_b.x)
                    
                    # Optimize a with b fixed
                    def neg_obj_a(a):
                        p_b = 1.0 - np.exp(-A * np.exp(-k * max(0, b_curr)) * dt)
                        p_a = 1.0 - np.exp(-A * np.exp(-k * max(0, a)) * dt)
                        val = ((1-p_b)*(1-p_a)*V_next_q 
                               + p_b*(1-p_a)*(b_curr + V_next_q_plus)
                               + (1-p_b)*p_a*(max(0,a) + V_next_q_minus)
                               + p_b*p_a*(b_curr + max(0,a) + V_next_q)
                               - hold_cost[q_idx])
                        return -val
                    
                    res_a = minimize_scalar(neg_obj_a, bounds=(0, max_depth), method='bounded', options={'xatol': 1e-6})
                    a_curr = max(0, res_a.x)
                
                best_bid_depth = b_curr
                best_ask_depth = a_curr
                best_val = -neg_obj_b(b_curr) if hasattr(neg_obj_b, '__call__') else 0.0
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
                def neg_obj_b(b):
                    p_b = 1.0 - np.exp(-A * np.exp(-k * max(0, b)) * dt)
                    val = (1-p_b)*V_next_q + p_b*(max(0,b) + V_next_q_plus) - hold_cost[q_idx]
                    return -val
                
                res = minimize_scalar(neg_obj_b, bounds=(0, max_depth), method='bounded', options={'xatol': 1e-6})
                best_bid_depth = max(0, res.x)
                best_ask_depth = 0.0
                best_val = -neg_obj_b(best_bid_depth)
                
            elif can_ask:
                # Only ask side active
                def neg_obj_a(a):
                    p_a = 1.0 - np.exp(-A * np.exp(-k * max(0, a)) * dt)
                    val = (1-p_a)*V_next_q + p_a*(max(0,a) + V_next_q_minus) - hold_cost[q_idx]
                    return -val
                
                res = minimize_scalar(neg_obj_a, bounds=(0, max_depth), method='bounded', options={'xatol': 1e-6})
                best_ask_depth = max(0, res.x)
                best_bid_depth = 0.0
                best_val = -neg_obj_a(best_ask_depth)
                
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


def _clip_depths(depths, min_d=0.0, max_d=np.inf):
    """Clip depths to valid range."""
    return np.clip(depths, min_d, max_d)


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
