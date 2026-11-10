"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths using continuous-depth optimization.
    Uses a fine grid with local refinement around the first-order condition optimum.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Pre-compute inventory costs
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Two-stage grid: coarse then fine
    # Coarse grid
    depth_grid_coarse = np.linspace(0.005, 5.0, 100)
    n_depths_coarse = len(depth_grid_coarse)
    
    # Pre-compute fill probabilities for coarse
    with np.errstate(over='ignore', under='ignore'):
        exp_term_c = np.exp(-k * depth_grid_coarse)
        p_depths_c = 1.0 - np.exp(-A * exp_term_c * dt)
    p_depths_c = np.clip(p_depths_c, 0.0, 1.0)
    
    # Build matrices for coarse
    P_b_c = np.tile(p_depths_c[:, np.newaxis], (1, n_depths_coarse))
    P_a_c = np.tile(p_depths_c[np.newaxis, :], (n_depths_coarse, 1))
    
    Depths_b_c = np.tile(depth_grid_coarse[:, np.newaxis], (1, n_depths_coarse))
    Depths_a_c = np.tile(depth_grid_coarse[np.newaxis, :], (n_depths_coarse, 1))
    Rewards_c = Depths_b_c * P_b_c + Depths_a_c * P_a_c
    
    P_plus_c = P_b_c * (1.0 - P_a_c)
    P_minus_c = P_a_c * (1.0 - P_b_c)
    P_same_c = 1.0 - P_plus_c - P_minus_c
    
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin
    
    # Phase 1: Coarse DP to get approximate optimal depths
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break
            
        V_next = V[n+1]
        
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        
        V_q_plus[:n_q-1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        Values = Rewards_c[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        Values += P_plus_c[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus_c[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same_c[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]
        
        Values[-1, :, :] -= P_plus_c * V_q_plus[-1, np.newaxis, np.newaxis]
        Values[0, :, :] -= P_minus_c * V_q_minus[0, np.newaxis, np.newaxis]
        
        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (n_depths_coarse, n_depths_coarse))
        
        V[n, :] = flat_values[np.arange(n_q), max_indices]
        opt_bid[n, :] = depth_grid_coarse[i_idx]
        opt_ask[n, :] = depth_grid_coarse[j_idx]
    
    # Phase 2: Fine grid around coarse optima
    # For each (n, q), create a local fine grid around the coarse optimum
    # We'll do this in batches to save memory
    
    # Determine fine grid range
    fine_radius = 0.5  # +/- 0.5 around coarse optimum
    n_fine = 50
    
    # Pre-allocate fine policy arrays
    opt_bid_fine = np.zeros((steps, n_q))
    opt_ask_fine = np.zeros((steps, n_q))
    
    # Process in chunks of steps to manage memory
    chunk_size = 100
    
    for n_start in range(0, steps, chunk_size):
        n_end = min(n_start + chunk_size, steps)
        
        # Collect coarse optima for this chunk
        coarse_bids = opt_bid[n_start:n_end]  # (chunk, n_q)
        coarse_asks = opt_ask[n_start:n_end]
        
        # Create fine grids for each (n, q) in this chunk
        # We'll process one step at a time within the chunk
        
        for n_local in range(n_end - n_start):
            n = n_start + n_local
            
            if time.time() > deadline:
                # Fill remaining with coarse values
                for n_fill in range(n, steps):
                    opt_bid_fine[n_fill] = opt_bid[n_fill]
                    opt_ask_fine[n_fill] = opt_ask[n_fill]
                break
            
            # For this step, we need V[n+1] from the coarse DP
            # But we want to recompute with fine grid using the same V[n+1]
            # Actually, we should do a second DP pass with fine grids
            
            # For now, just interpolate fine depths around coarse optima
            # This is a simplification; a proper second DP would be better
            
            q_idx = np.arange(n_q)
            
            # Create fine grid for each q
            for q in range(n_q):
                b_center = coarse_bids[n_local, q]
                a_center = coarse_asks[n_local, q]
                
                # Create local fine grids
                b_fine = np.linspace(max(0.001, b_center - fine_radius), 
                                     min(10.0, b_center + fine_radius), n_fine)
                a_fine = np.linspace(max(0.001, a_center - fine_radius), 
                                     min(10.0, a_center + fine_radius), n_fine)
                
                # Evaluate on fine grid
                with np.errstate(over='ignore', under='ignore'):
                    exp_b = np.exp(-k * b_fine)
                    exp_a = np.exp(-k * a_fine)
                    p_b = 1.0 - np.exp(-A * exp_b * dt)
                    p_a = 1.0 - np.exp(-A * exp_a * dt)
                p_b = np.clip(p_b, 0.0, 1.0)
                p_a = np.clip(p_a, 0.0, 1.0)
                
                # Build matrices
                P_b_f = np.tile(p_b[:, np.newaxis], (1, n_fine))
                P_a_f = np.tile(p_a[np.newaxis, :], (n_fine, 1))
                
                Depths_b_f = np.tile(b_fine[:, np.newaxis], (1, n_fine))
                Depths_a_f = np.tile(a_fine[np.newaxis, :], (n_fine, 1))
                Rewards_f = Depths_b_f * P_b_f + Depths_a_f * P_a_f
                
                P_plus_f = P_b_f * (1.0 - P_a_f)
                P_minus_f = P_a_f * (1.0 - P_b_f)
                P_same_f = 1.0 - P_plus_f - P_minus_f
                
                # Get V values for this q
                q_idx = q
                V_q_plus_val = V[n+1, min(q+1, n_q-1)]
                V_q_minus_val = V[n+1, max(q-1, 0)]
                V_q_val = V[n+1, q]
                
                # Compute values
                Values_f = Rewards_f - C_inv[q]
                Values_f += P_plus_f * V_q_plus_val
                Values_f += P_minus_f * V_q_minus_val
                Values_f += P_same_f * V_q_val
                
                # Boundary conditions
                if q == n_q - 1:  # q = Q
                    Values_f -= P_plus_f * V_q_plus_val
                if q == 0:  # q = -Q
                    Values_f -= P_minus_f * V_q_minus_val
                
                flat_f = Values_f.reshape(-1)
                max_idx = np.argmax(flat_f)
                i_f, j_f = np.unravel_index(max_idx, (n_fine, n_fine))
                
                opt_bid_fine[n, q] = b_fine[i_f]
                opt_ask_fine[n, q] = a_fine[j_f]
        
        else:
            continue
        break
    
    # Use fine policy if computed, else coarse
    # Check if fine policy was fully computed
    if np.all(opt_bid_fine > 0) and np.all(opt_ask_fine > 0):
        return opt_bid_fine, opt_ask_fine
    else:
        return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q']
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
