"""Optimal market making with inventory risk - MODEL-AND-OPTIMIZE approach.

Uses coarse grid + iterative coordinate descent + high-precision polish.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill

def _fill_prob(depth, A, k, dt):
    """Fill probability for a given depth."""
    if depth <= 0:
        return 0.0
    exp_term = np.exp(-k * depth)
    p = 1.0 - np.exp(-A * exp_term * dt)
    return min(max(p, 0.0), 1.0)

def _fill_prob_vec(depths, A, k, dt):
    """Vectorized fill probability."""
    depths = np.asarray(depths, dtype=float)
    p = np.where(depths > 0, 1.0 - np.exp(-A * np.exp(-k * depths) * dt), 0.0)
    return np.clip(p, 0.0, 1.0)

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths using:
    1. Coarse grid for initial optima
    2. Iterative coordinate descent for joint refinement
    3. High-precision local search for final polish
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Finer coarse grid
    coarse_grid = np.linspace(0.01, 4.0, 200)
    n_coarse = len(coarse_grid)
    
    # V[n, q_idx]
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))
    
    # Pre-compute fill probabilities on coarse grid
    p_coarse = _fill_prob_vec(coarse_grid, A, k, dt)
    
    # Matrix for coarse grid
    P_b = np.tile(p_coarse[:, np.newaxis], (1, n_coarse))
    P_a = np.tile(p_coarse[np.newaxis, :], (n_coarse, 1))
    
    Depths_b = np.tile(coarse_grid[:, np.newaxis], (1, n_coarse))
    Depths_a = np.tile(coarse_grid[np.newaxis, :], (n_coarse, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus
    
    # Number of coordinate descent iterations
    n_cd_iters = 5
    # Points for local refinement
    n_local = 50
    
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break
        
        V_next = V[n + 1]
        
        # Transition inventory offsets
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        
        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        # Compute values on coarse grid
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]
        
        # Boundary handling
        Values[-1, :, :] -= P_plus * V_q_plus[-1, np.newaxis, np.newaxis]
        Values[0, :, :] -= P_minus * V_q_minus[0, np.newaxis, np.newaxis]
        
        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (n_coarse, n_coarse))
        
        b_coarse = coarse_grid[i_idx]
        a_coarse = coarse_grid[j_idx]
        
        # Coordinate descent + local refinement for each q
        refined_bid = np.zeros(n_q)
        refined_ask = np.zeros(n_q)
        V_refined = np.zeros(n_q)
        
        for q_idx in range(n_q):
            b_cur = b_coarse[q_idx]
            a_cur = a_coarse[q_idx]
            
            # Iterative coordinate descent
            for cd_iter in range(n_cd_iters):
                # Optimize bid given ask
                if q_idx == n_q - 1:
                    b_opt = b_cur  # bid doesn't matter at q=Q
                else:
                    if b_cur < 0.05:
                        b_range = np.linspace(0.005, 0.4, n_local)
                    elif b_cur > 3.9:
                        b_range = np.linspace(3.5, 5.0, n_local)
                    else:
                        b_range = np.linspace(max(0.005, b_cur - 0.4), min(5.0, b_cur + 0.4), n_local)
                    
                    p_b_arr = _fill_prob_vec(b_range, A, k, dt)
                    p_a0 = _fill_prob(a_cur, A, k, dt)
                    
                    if q_idx == 0:
                        trans = p_b_arr * V_q_plus[q_idx] + (1 - p_b_arr) * V_q[q_idx]
                        val_b = b_range * p_b_arr - C_inv[q_idx] + trans
                    else:
                        trans = p_b_arr * (1 - p_a0) * V_q_plus[q_idx] + \
                                p_a0 * (1 - p_b_arr) * V_q_minus[q_idx] + \
                                (1 - p_b_arr - p_a0 + p_b_arr * p_a0) * V_q[q_idx]
                        val_b = b_range * p_b_arr + a_cur * p_a0 - C_inv[q_idx] + trans
                    
                    b_opt = b_range[np.argmax(val_b)]
                
                # Optimize ask given bid
                if q_idx == 0:
                    a_opt = a_cur  # ask doesn't matter at q=-Q
                else:
                    if a_cur < 0.05:
                        a_range = np.linspace(0.005, 0.4, n_local)
                    elif a_cur > 3.9:
                        a_range = np.linspace(3.5, 5.0, n_local)
                    else:
                        a_range = np.linspace(max(0.005, a_cur - 0.4), min(5.0, a_cur + 0.4), n_local)
                    
                    p_a_arr = _fill_prob_vec(a_range, A, k, dt)
                    p_bopt = _fill_prob(b_opt, A, k, dt)
                    
                    if q_idx == n_q - 1:
                        trans = p_a_arr * V_q_minus[q_idx] + (1 - p_a_arr) * V_q[q_idx]
                        val_a = a_range * p_a_arr - C_inv[q_idx] + trans
                    else:
                        trans = p_bopt * (1 - p_a_arr) * V_q_plus[q_idx] + \
                                p_a_arr * (1 - p_bopt) * V_q_minus[q_idx] + \
                                (1 - p_bopt - p_a_arr + p_bopt * p_a_arr) * V_q[q_idx]
                        val_a = b_opt * p_bopt + a_range * p_a_arr - C_inv[q_idx] + trans
                    
                    a_opt = a_range[np.argmax(val_a)]
                
                b_cur = b_opt
                a_cur = a_opt
            
            refined_bid[q_idx] = b_cur
            refined_ask[q_idx] = a_cur
            
            # Compute refined value
            pb = _fill_prob(b_cur, A, k, dt)
            pa = _fill_prob(a_cur, A, k, dt)
            
            if q_idx == n_q - 1:
                trans = pa * V_q_minus[q_idx] + (1 - pa) * V_q[q_idx]
                V_refined[q_idx] = a_cur * pa - C_inv[q_idx] + trans
            elif q_idx == 0:
                trans = pb * V_q_plus[q_idx] + (1 - pb) * V_q[q_idx]
                V_refined[q_idx] = b_cur * pb - C_inv[q_idx] + trans
            else:
                trans = pb * (1 - pa) * V_q_plus[q_idx] + \
                        pa * (1 - pb) * V_q_minus[q_idx] + \
                        (1 - pb - pa + pb * pa) * V_q[q_idx]
                V_refined[q_idx] = b_cur * pb + a_cur * pa - C_inv[q_idx] + trans
        
        V[n, :] = V_refined
        opt_bid[n, :] = refined_bid
        opt_ask[n, :] = refined_ask
    
    return opt_bid, opt_ask


def _get_policy(params):
    """Retrieves or computes the optimal policy for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Returns optimal bid and ask depths based on pre-computed DP policy."""
    opt_bid, opt_ask = _get_policy(params)
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1
    
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Ensure valid outputs
    bid_depths = np.where(np.isnan(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)
    
    return bid_depths.astype(float), ask_depths.astype(float)
