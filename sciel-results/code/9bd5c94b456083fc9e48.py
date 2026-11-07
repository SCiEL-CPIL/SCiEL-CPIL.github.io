"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np

_PARAMS_CACHE = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function: V_next[q_idx] = optimal expected objective from next step with inventory q
    # V at terminal = 0 (no terminal charge)
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Depth grid: use a logarithmic grid from small to large depths
    # The fill probability is p = 1 - exp(-A * exp(-k*d) * dt)
    # For very small d, p approaches 1; for very large d, p approaches 0
    # We want to capture the range where the optimal depth might lie
    # Expanded range and higher resolution for better optimality
    min_depth = 1e-4
    max_depth = 10.0 / k  # Ensure fill prob is negligible at max depth
    n_depths = 100
    depth_grid = np.geomspace(min_depth, max_depth, n_depths)
    
    # Pre-compute fill probabilities for each depth
    # p[d] = 1 - exp(-A * exp(-k*d) * dt)
    # Use log-sum-exp or careful calculation to avoid underflow/overflow if possible, 
    # but with these parameters, direct exp is usually fine.
    # A * exp(-k*d) * dt can be very small, so exp(-small) ~ 1 - small.
    # p = 1 - exp(-lambda) ~ lambda for small lambda.
    # lambda = A * exp(-k*d) * dt
    lambdas = A * np.exp(-k * depth_grid) * dt
    # For very small lambdas, 1 - exp(-lambdas) suffers from cancellation.
    # Use -expm1(-lambdas) for better precision.
    fill_probs = -np.expm1(-lambdas)
    
    # Inventory risk cost coefficient
    risk_coeff = 0.5 * gamma * sigma**2 * dt
    
    for n in range(steps - 1, -1, -1):
        for q_idx in range(n_inv):
            q = q_idx - Q  # inventory level
            
            best_val = -np.inf
            best_bid_d = depth_grid[0]
            best_ask_d = depth_grid[0]
            
            # Risk cost for this step (incurred regardless of fills)
            risk_cost = risk_coeff * q * q
            
            # Try all combinations of bid and ask depths
            # To speed up, we can exploit the structure:
            # E = -risk_cost + p_b*p_a*(b+a+V_next[q_idx]) + p_b*(1-p_a)*(b+V_next[q_idx+1]) 
            #     + (1-p_b)*p_a*(a+V_next[q_idx-1]) + (1-p_b)*(1-p_a)*V_next[q_idx]
            # 
            # For fixed ask depth a, the optimal bid depth can be found by 1D search
            # and vice versa. But for simplicity with 40x40=1600 combos per (n,q),
            # total is 1000 * 61 * 1600 = ~97.6M operations which might be too slow.
            # Let's use a coarser grid: 25 depths -> 625 combos -> 1000*61*625 = ~38M
            # Still might be slow in Python. Let's vectorize over depths.
            
            # Vectorized approach: for each bid depth b, compute the best ask depth
            # Actually, let's just do a 2D vectorized computation
            
            # V_next indices
            V_q = V_next[q_idx]
            V_q1 = V_next[min(q_idx + 1, n_inv - 1)] if q < Q else 0.0  # no bid fill at q=Q
            V_qm1 = V_next[max(q_idx - 1, 0)] if q > -Q else 0.0  # no ask fill at q=-Q
            
            # If q == Q, bid is disabled; if q == -Q, ask is disabled
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                # No quotes possible
                bid_table[n, q_idx] = np.inf
                ask_table[n, q_idx] = np.inf
                V_next[q_idx] = -risk_cost + V_q
                continue
            
            if can_bid and can_ask:
                # Both sides active: 2D optimization
                # For efficiency, iterate over bid depths and for each, find best ask depth
                for bi, b in enumerate(depth_grid):
                    pb = fill_probs[bi]
                    # Expected value as function of ask depth a:
                    # E(a) = -risk_cost + pb*pa*(b+a+V_q) + pb*(1-pa)*(b+V_q1) 
                    #        + (1-pb)*pa*(a+V_qm1) + (1-pb)*(1-pa)*V_q
                    # = -risk_cost + pb*b*(pa+1-pa) + pa*(1-pb)*a + pa*(1-pb)*V_qm1
                    #   + pb*(1-pa)*V_q1 + (1-pb)*(1-pa)*V_q + pa*(1-pb)*a ... let me redo
                    # 
                    # = -risk_cost + pb*b + pa*a*(pb + 1 - pb) ... no
                    # Let me expand carefully:
                    # pb*pa*(b+a+V_q) + pb*(1-pa)*(b+V_q1) + (1-pb)*pa*(a+V_qm1) + (1-pb)*(1-pa)*V_q
                    # = pb*pa*b + pb*pa*a + pb*pa*V_q + pb*b - pb*pa*b + pb*V_q1 - pb*pa*V_q1
                    #   + pa*a - pb*pa*a + pa*V_qm1 - pb*pa*V_qm1 + V_q - pb*V_q - pa*V_q + pb*pa*V_q
                    # = pb*b + pa*a + pb*V_q1 - pb*pa*V_q1 + pa*V_qm1 - pb*pa*V_qm1 + V_q - pb*V_q - pa*V_q
                    # = pb*b + pa*a + V_q*(1 - pb - pa + pb*pa) + pb*(1-pa)*V_q1 + pa*(1-pb)*V_qm1
                    # = pb*b + pa*a + (1-pb)*(1-pa)*V_q + pb*(1-pa)*V_q1 + pa*(1-pb)*V_qm1
                    
                    # For fixed b (pb), this is linear in pa and a:
                    # E = pb*b + (1-pb)*V_q + pb*(1-pa)*V_q1 + pa*(1-pb)*V_qm1 + pa*a
                    # = pb*b + (1-pb)*V_q + pb*V_q1 - pb*pa*V_q1 + pa*V_qm1 - pb*pa*V_qm1 + pa*a
                    # = pb*b + (1-pb)*V_q + pb*V_q1 + pa*(V_qm1 - pb*V_q1 - pb*V_qm1 + a)
                    # = pb*b + (1-pb)*V_q + pb*V_q1 + pa*(a + V_qm1 - pb*(V_q1 + V_qm1))
                    
                    # So for fixed b, E is linear in pa (which is monotonic in a).
                    # The coefficient of pa is: a + V_qm1 - pb*(V_q1 + V_qm1)
                    # If this coefficient > 0, we want max pa (min a); if < 0, we want min pa (max a)
                    # But a and pa are linked: pa = 1 - exp(-A*exp(-k*a)*dt), monotonic in a
                    
                    # This is getting complex. Let's just do the full 2D grid with vectorization.
                    pass
                
                # Vectorized 2D: compute all (bi, ai) combinations
                # pb[bi], pa[ai], b[bi], a[ai]
                PB = fill_probs[:, np.newaxis]  # (n_depths, 1)
                PA = fill_probs[np.newaxis, :]  # (1, n_depths)
                B = depth_grid[:, np.newaxis]  # (n_depths, 1)
                Aa = depth_grid[np.newaxis, :]  # (1, n_depths)
                
                # E = -risk_cost + PB*PA*(B+Aa+V_q) + PB*(1-PA)*(B+V_q1) + (1-PB)*PA*(Aa+V_qm1) + (1-PB)*(1-PA)*V_q
                E = (-risk_cost 
                     + PB * PA * (B + Aa + V_q) 
                     + PB * (1 - PA) * (B + V_q1) 
                     + (1 - PB) * PA * (Aa + V_qm1) 
                     + (1 - PB) * (1 - PA) * V_q)
                
                max_idx = np.argmax(E)
                bi_best = max_idx // n_depths
                ai_best = max_idx % n_depths
                
                best_bid_d = depth_grid[bi_best]
                best_ask_d = depth_grid[ai_best]
                bid_table[n, q_idx] = best_bid_d
                ask_table[n, q_idx] = best_ask_d
                V_next[q_idx] = E[bi_best, ai_best]
                
            elif can_bid:
                # Only bid active (q == -Q, no ask)
                # Vectorized 1D search
                # E(bi) = -risk_cost + (1 - pb[bi])*V_q + pb[bi]*(depth_grid[bi] + V_q1)
                vals = -risk_cost + (1 - fill_probs) * V_q + fill_probs * (depth_grid + V_q1)
                best_idx = np.argmax(vals)
                best_bid_d = depth_grid[best_idx]
                best_val = vals[best_idx]
                
                bid_table[n, q_idx] = best_bid_d
                ask_table[n, q_idx] = np.inf
                V_next[q_idx] = best_val
                
            else:
                # Only ask active (q == Q, no bid)
                # Vectorized 1D search
                # E(ai) = -risk_cost + (1 - pa[ai])*V_q + pa[ai]*(depth_grid[ai] + V_qm1)
                vals = -risk_cost + (1 - fill_probs) * V_q + fill_probs * (depth_grid + V_qm1)
                best_idx = np.argmax(vals)
                best_ask_d = depth_grid[best_idx]
                best_val = vals[best_idx]
                
                bid_table[n, q_idx] = np.inf
                ask_table[n, q_idx] = best_ask_d
                V_next[q_idx] = best_val
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _PARAMS_CACHE:
        _PARAMS_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], 
            params['gamma'], params['T'], params['steps'], params['Q']
        )
    return _PARAMS_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote optimal bid/ask depths based on pre-computed policy table.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    bid_table, ask_table = _get_policy_table(params)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    T = params['T']
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to table indices
    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(int)
    
    # Get depths from table
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # Ensure no bid at q=Q and no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure depths are valid (positive, not too small, not NaN)
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, 1e-6), bid_depths)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, 1e-6), ask_depths)
    
    return bid_depths, ask_depths
