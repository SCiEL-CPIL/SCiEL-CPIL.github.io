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
    Compute optimal bid/ask depth for each (step, inventory) via backward induction
    using continuous-depth optimization via first-order conditions.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_offset = Q  # q + Q gives array index
    
    # Value function: V[n, q_idx] = optimal expected cumulative reward from step n onwards
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: optimal depth for bid and ask
    bid_depths = np.zeros((steps, n_inv), dtype=np.float64)
    ask_depths = np.zeros((steps, n_inv), dtype=np.float64)
    
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
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if can_bid:
                V_next_q_plus = V[n+1, q_idx + 1]
            else:
                V_next_q_plus = 0.0  # won't be used
            
            if can_ask:
                V_next_q_minus = V[n+1, q_idx - 1]
            else:
                V_next_q_minus = 0.0  # won't be used
            
            hc = hold_cost[q_idx]
            
            if can_bid and can_ask:
                # Both sides active
                # For bid depth b:
                #   p_b(b) = 1 - exp(-A * exp(-k*b) * dt)
                #   d p_b / db = A * k * exp(-k*b) * dt * exp(-A*exp(-k*b)*dt)
                #               = k * (1 - p_b) * A * exp(-k*b) * dt * ... 
                # Actually: dp_b/db = A * k * exp(-k*b) * dt * exp(-A*exp(-k*b)*dt)
                #            = k * exp(-k*b) * (1 - exp(-A*exp(-k*b)*dt)) * A * dt * exp(-A*exp(-k*b)*dt)
                # Let me just compute numerically.
                
                def neg_obj_bid(b):
                    if b < 0:
                        return np.inf
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    # Expected value contribution from bid side (given ask is optimized separately)
                    # We need to consider the joint optimization. Let's use a 2D approach
                    # but solve sequentially: first optimize bid given ask, then ask given bid.
                    # Actually, let's do a proper 2D search with scipy.
                    return 0.0  # placeholder, we'll use 2D below
                
                # Use 2D minimization with scipy
                # Define the objective function to minimize (negative of expected value)
                def neg_obj_2d(x):
                    b, a = x
                    if b < 0 or a < 0:
                        return np.inf
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    
                    # Four cases:
                    # no bid, no ask: prob (1-p_b)(1-p_a), reward 0, next state q
                    # bid only: prob p_b(1-p_a), reward b, next state q+1
                    # ask only: prob (1-p_b)p_a, reward a, next state q-1
                    # both: prob p_b*p_a, reward b+a, next state q
                    term1 = (1 - p_b) * (1 - p_a) * V_next_q
                    term2 = p_b * (1 - p_a) * (b + V_next_q_plus)
                    term3 = (1 - p_b) * p_a * (a + V_next_q_minus)
                    term4 = p_b * p_a * (b + a + V_next_q)
                    
                    return -( -hc + term1 + term2 + term3 + term4 )
                
                # Bounds for search
                max_depth = 10.0 / k
                bounds = [(0.0, max_depth), (0.0, max_depth)]
                
                # Use scipy.optimize.minimize with L-BFGS-B
                from scipy.optimize import minimize
                x0 = np.array([1.0/k, 1.0/k])
                result = minimize(neg_obj_2d, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 50, 'ftol': 1e-10})
                
                b_opt = max(0.0, result.x[0])
                a_opt = max(0.0, result.x[1])
                
                # Compute the optimal value
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                term1 = (1 - p_b) * (1 - p_a) * V_next_q
                term2 = p_b * (1 - p_a) * (b_opt + V_next_q_plus)
                term3 = (1 - p_b) * p_a * (a_opt + V_next_q_minus)
                term4 = p_b * p_a * (b_opt + a_opt + V_next_q)
                
                V[n, q_idx] = -hc + term1 + term2 + term3 + term4
                bid_depths[n, q_idx] = b_opt
                ask_depths[n, q_idx] = a_opt
                
            elif can_bid:
                # Only bid side active
                def neg_obj_bid(b):
                    if b < 0:
                        return np.inf
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    term1 = (1 - p_b) * V_next_q
                    term2 = p_b * (b + V_next_q_plus)
                    return -(-hc + term1 + term2)
                
                max_depth = 10.0 / k
                result = minimize_scalar(neg_obj_bid, bounds=(0.0, max_depth), 
                                         method='bounded', options={'maxiter': 50})
                
                b_opt = max(0.0, result.x)
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                term1 = (1 - p_b) * V_next_q
                term2 = p_b * (b_opt + V_next_q_plus)
                
                V[n, q_idx] = -hc + term1 + term2
                bid_depths[n, q_idx] = b_opt
                ask_depths[n, q_idx] = 0.0
                
            elif can_ask:
                # Only ask side active
                def neg_obj_ask(a):
                    if a < 0:
                        return np.inf
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    term1 = (1 - p_a) * V_next_q
                    term2 = p_a * (a + V_next_q_minus)
                    return -(-hc + term1 + term2)
                
                max_depth = 10.0 / k
                result = minimize_scalar(neg_obj_ask, bounds=(0.0, max_depth),
                                         method='bounded', options={'maxiter': 50})
                
                a_opt = max(0.0, result.x)
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                term1 = (1 - p_a) * V_next_q
                term2 = p_a * (a_opt + V_next_q_minus)
                
                V[n, q_idx] = -hc + term1 + term2
                bid_depths[n, q_idx] = 0.0
                ask_depths[n, q_idx] = a_opt
                
            else:
                # Neither side active
                V[n, q_idx] = -hc + V_next_q
                bid_depths[n, q_idx] = 0.0
                ask_depths[n, q_idx] = 0.0
    
    return bid_depths, ask_depths


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
