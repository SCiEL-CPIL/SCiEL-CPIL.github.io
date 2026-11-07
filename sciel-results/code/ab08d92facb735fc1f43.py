"""Optimal market making with inventory risk - DP precomputation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precomputed optimal policy cache
_policy_cache = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """Solve the DP problem and return optimal bid/ask depth tables."""
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # Depth grid: log-spaced for finer resolution at small depths
    # Depths must be >= 0; use np.inf for "no quote"
    # Grid from small to large, plus inf option
    n_depths = 120
    # Log-spaced grid
    d_min = 0.001
    d_max = 5.0 / k
    depth_grid = np.geomspace(d_min, d_max, n_depths)
    # Include 0 (no quote equivalent, very small depth ~ 0) and inf
    # Actually depth=0 means quote at mid, which is valid
    # Let's include a range: very small to large
    depth_grid = np.concatenate([[0.0], depth_grid])
    n_d = len(depth_grid)
    
    # Fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    # For d=0: p = 1 - exp(-A * dt)
    # For large d: p -> 0
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[t, inv] where inv in 0..n_inv-1 corresponds to q = -Q + inv
    # V[n_steps, q] = 0 for all q (no terminal charge)
    V = np.zeros((n_steps + 1, n_inv))
    
    # Optimal depth indices
    opt_bid = np.zeros((n_steps, n_inv), dtype=np.int32)
    opt_ask = np.zeros((n_steps, n_inv), dtype=np.int32)
    
    # Backward induction
    for t in range(n_steps - 1, -1, -1):
        V_next = V[t + 1]
        # Inventory cost for this step
        q_vals = np.arange(-Q, Q + 1)  # shape (n_inv,)
        cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt  # shape (n_inv,)
        
        # For each inventory level, we need to choose (bid_d, ask_d) to maximize:
        # - cost + E[depth_earned] + V_next[new_inv]
        # 
        # Bid fill at inv i (q = -Q + i): only if q < Q, i.e., i < n_inv - 1
        # Ask fill at inv i: only if q > -Q, i.e., i > 0
        #
        # Joint: both can fill independently
        # P(bid fill) = p_b, P(ask fill) = p_a, independent
        # 
        # Expected value for given (bid_idx, ask_idx):
        # p_b * (depth_bid + V_next[i+1]) + p_a * (-depth_ask + V_next[i-1]) 
        # + (1-p_b)(1-p_a) * V_next[i]
        # - p_b * p_a * (0 + V_next[i])  # both fill, net depth 0, inv unchanged
        # 
        # Wait, let me redo:
        # If bid fills: earn +depth_bid, inv increases by 1
        # If ask fills: earn -depth_ask, inv decreases by 1
        # If both: earn +depth_bid - depth_ask, inv unchanged
        # If neither: earn 0, inv unchanged
        #
        # E[reward] = p_b*(1-p_a)*depth_bid + p_a*(1-p_b)*(-depth_ask) + p_b*p_a*(depth_bid - depth_ask)
        #           = p_b*depth_bid - p_a*depth_ask
        #
        # E[V_next] = (1-p_b)*(1-p_a)*V_next[i] + p_b*(1-p_a)*V_next[i+1] 
        #           + (1-p_b)*p_a*V_next[i-1] + p_b*p_a*V_next[i]
        #           = (1-p_a)*V_next[i] + p_a*V_next[i-1] - p_b*p_a*V_next[i] + p_b*(1-p_a)*V_next[i+1] + p_b*p_a*V_next[i]
        # Let me just compute directly:
        # = (1-p_b)*(1-p_a)*V[i] + p_b*(1-p_a)*V[i+1] + (1-p_b)*p_a*V[i-1] + p_b*p_a*V[i]
        # = (1-p_a)*V[i] + p_a*V[i-1] + p_b*(1-p_a)*V[i+1] + p_b*p_a*V[i] - p_b*(1-p_a)*V[i]
        # Hmm, let me just keep it as the 4-term sum.
        
        # Vectorize over inventory
        # For each inv i, compute the value for all (bid_idx, ask_idx) pairs
        # This is n_d x n_d per inv, which is 121 x 121 = 14641 per inv, times 61 inv = ~893k per step
        # Times 1000 steps = ~893M operations... might be too slow.
        
        # Let's reduce: use fewer depths or exploit structure.
        # Actually, let's try with n_d = 80 first.
        
        V_t = np.zeros(n_inv)
        for i in range(n_inv):
            q = -Q + i
            # Boundary conditions
            can_bid = (q < Q)  # can have bid fill
            can_ask = (q > -Q)  # can have ask fill
            
            if not can_bid and not can_ask:
                # At boundaries where no fills possible (shouldn't happen for -Q < q < Q)
                V_t[i] = -cost[i] + V_next[i]
                opt_bid[t, i] = 0
                opt_ask[t, i] = 0
                continue
            
            # Compute expected next value for each (bid, ask) combination
            # p_b depends on bid_idx, p_a depends on ask_idx
            # Only consider valid indices
            
            if can_bid:
                pb = p_fill  # shape (n_d,)
                db = depth_grid  # shape (n_d,)
                # If bid fills, inv goes to i+1
                V_bid = V_next[i + 1]
            else:
                pb = np.array([0.0])
                db = np.array([0.0])
                V_bid = 0.0
            
            if can_ask:
                pa = p_fill  # shape (n_d,)
                da = depth_grid  # shape (n_d,)
                V_ask = V_next[i - 1]
            else:
                pa = np.array([0.0])
                da = np.array([0.0])
                V_ask = 0.0
            
            n_pb = len(pb)
            n_pa = len(pa)
            
            # For each (bid_idx, ask_idx), compute total value
            # pb_b = pb[bid_idx], pa_a = pa[ask_idx]
            # db_b = db[bid_idx], da_a = da[ask_idx]
            # 
            # E[depth] = pb_b * db_b - pa_a * da_a
            # E[V_next] = (1-pb_b)*(1-pa_a)*V_next[i] + pb_b*(1-pa_a)*V_bid 
            #            + (1-pb_b)*pa_a*V_ask + pb_b*pa_a*V_next[i]
            
            # Vectorize: create matrices
            # pb_mat: (n_pb, 1), pa_mat: (1, n_pa)
            pb_mat = pb[:, np.newaxis]  # (n_pb, 1)
            pa_mat = pa[np.newaxis, :]  # (1, n_pa)
            db_mat = db[:, np.newaxis]  # (n_pb, 1)
            da_mat = da[np.newaxis, :]  # (1, n_pa)
            
            E_depth = pb_mat * db_mat - pa_mat * da_mat  # (n_pb, n_pa)
            
            E_Vnext = ((1 - pb_mat) * (1 - pa_mat) * V_next[i] 
                      + pb_mat * (1 - pa_mat) * V_bid
                      + (1 - pb_mat) * pa_mat * V_ask
                      + pb_mat * pa_mat * V_next[i])  # (n_pb, n_pa)
            
            total = E_depth + E_Vnext - cost[i]  # (n_pb, n_pa)
            
            # Find optimal (bid_idx, ask_idx)
            idx = np.unravel_index(np.argmax(total), total.shape)
            opt_bid[t, i] = idx[0]
            opt_ask[t, i] = idx[1]
            V_t[i] = total[idx]
        
        V[t] = V_t
    
    # Convert optimal indices to depths
    bid_depths_table = depth_grid[opt_bid]  # shape (n_steps, n_inv)
    ask_depths_table = depth_grid[opt_ask]  # shape (n_steps, n_inv)
    
    return bid_depths_table, ask_depths_table, depth_grid


def _get_policy(params):
    """Get or compute the optimal policy for given params."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        _policy_cache[key] = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths."""
    t_start = time.time()
    deadline = t_start + 290  # Safety margin before 300s limit
    
    try:
        bid_table, ask_table, depth_grid = _get_policy(params)
        
        steps = params['steps']
        Q = params['Q']
        
        # Determine current time step
        dt = params['T'] / steps
        t_step = int(round(t / dt))
        t_step = max(0, min(t_step, steps - 1))
        
        # Map inventory to index: inv_idx = q + Q
        inv_idx = inventory + Q
        inv_idx = np.clip(inv_idx, 0, 2 * Q)
        
        # Look up optimal depths
        bid_depths = bid_table[t_step, inv_idx]
        ask_depths = ask_table[t_step, inv_idx]
        
        # Ensure valid: depths must be >= 0 and finite or inf
        # Values below -3/k make result invalid, so ensure all >= 0
        bid_depths = np.maximum(bid_depths, 0.0)
        ask_depths = np.maximum(ask_depths, 0.0)
        
        return bid_depths, ask_depths
    except Exception:
        # Fallback: risk-neutral depth
        k = params['k']
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
# EVOLVE-BLOCK-END
