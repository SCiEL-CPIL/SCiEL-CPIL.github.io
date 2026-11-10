"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 315.0  # Safety margin before 330s kill

def _compute_policy(sigma, A, k, gamma, T, steps, Q, depth_grid):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Uses vectorized DP with a refined depth grid and boundary handling.
    """
    dt = T / steps
    depths = depth_grid
    n_depths = len(depths)
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depths)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    P_b = np.tile(p_depths[:, np.newaxis], (1, n_depths))
    P_a = np.tile(p_depths[np.newaxis, :], (n_depths, 1))
    
    Depths_b = np.tile(depths[:, np.newaxis], (1, n_depths))
    Depths_a = np.tile(depths[np.newaxis, :], (n_depths, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus
    
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    mask_plus = np.ones((2*Q+1, 1, 1))
    mask_plus[-1, 0, 0] = 0.0
    
    mask_minus = np.ones((2*Q+1, 1, 1))
    mask_minus[0, 0, 0] = 0.0
    
    P_plus_masked = P_plus[np.newaxis, :, :] * mask_plus
    P_minus_masked = P_minus[np.newaxis, :, :] * mask_minus
    P_same_masked = (1.0 - P_plus_masked - P_minus_masked)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break
            
        V_next = V[n+1]
        
        V_q_plus = np.zeros((2*Q+1, 1, 1))
        V_q_minus = np.zeros((2*Q+1, 1, 1))
        V_q = np.zeros((2*Q+1, 1, 1))
        
        V_q[:, 0, 0] = V_next
        V_q_plus[:2*Q, 0, 0] = V_next[1:]
        V_q_minus[1:, 0, 0] = V_next[:-1]
        
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis] + \
                 P_plus_masked * V_q_plus + \
                 P_minus_masked * V_q_minus + \
                 P_same_masked * V_q
        
        flat_values = Values.reshape(2*Q+1, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (n_depths, n_depths))
        
        V[n, :] = flat_values[np.arange(2*Q+1), max_indices]
        opt_bid[n, :] = depths[i_idx]
        opt_ask[n, :] = depths[j_idx]
        
    return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Use a log-spaced grid for better resolution near small depths
    # where the optimal depth typically lies
    n_points = 80
    depth_grid = np.geomspace(0.001, 8.0, n_points)
    depth_grid = np.maximum(depth_grid, 0.001)
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q'], depth_grid
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    opt_bid, opt_ask = _get_policy(params)
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    min_depth = 0.001
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
