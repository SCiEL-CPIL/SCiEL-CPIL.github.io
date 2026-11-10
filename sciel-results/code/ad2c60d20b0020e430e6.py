"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 310.0  # Safety margin before 330s kill

def _compute_policy(sigma, A, k, gamma, T, steps, Q, depth_grid):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Uses vectorized DP with a refined depth grid and boundary handling.
    """
    dt = T / steps
    depths = depth_grid
    n_depths = len(depths)
    n_q = 2 * Q + 1
    
    # State space: q ranges from -Q to Q
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    # Pre-compute fill probabilities for each depth
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depths)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    # Pre-compute meshgrids for rewards and probabilities
    P_b = np.tile(p_depths[:, np.newaxis], (1, n_depths))
    P_a = np.tile(p_depths[np.newaxis, :], (n_depths, 1))
    
    Depths_b = np.tile(depths[:, np.newaxis], (1, n_depths))
    Depths_a = np.tile(depths[np.newaxis, :], (n_depths, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus
    
    # Pre-compute inventory costs
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fill remaining with last computed values
            for nn in range(n, -1, -1):
                V[nn, :] = V[n+1, :]
                opt_bid[nn, :] = opt_bid[n+1, :] if n+1 < steps else 0.0
                opt_ask[nn, :] = opt_ask[n+1, :] if n+1 < steps else 0.0
            break
            
        V_next = V[n+1]
        
        # Boundary handling
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        
        V_q_plus[:n_q-1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        # Compute values for all (i, j, q)
        # Shape: (n_q, n_depths, n_depths)
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        
        # Add transition terms
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]
        
        # Apply boundary masks: no bid fill at q=Q, no ask fill at q=-Q
        # At q=Q (last index), remove the bid fill contribution
        Values[-1, :, :] -= P_plus * V_q_plus[-1, np.newaxis, np.newaxis]
        # At q=-Q (first index), remove the ask fill contribution
        Values[0, :, :] -= P_minus * V_q_minus[0, np.newaxis, np.newaxis]
        
        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (n_depths, n_depths))
        
        V[n, :] = flat_values[np.arange(n_q), max_indices]
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
    
    # Depth grid: 150 points from 0.002 to 5.0
    # The optimal spread is typically around 1/k to a few units
    depth_grid = np.linspace(0.002, 5.0, 150)
    
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
    # Get the policy
    opt_bid, opt_ask = _get_policy(params)
    
    # t is current time. n is step index.
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # inventory is an array of ints
    Q = params['Q']
    q_idx = inventory + Q
    
    # Ensure indices are valid
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Return as float arrays
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
