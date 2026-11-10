"""Consumption-saving policy via vectorized backward induction with FOC refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    vectorized objective evaluation with FOC-based refinement.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Precompute utility and marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return np.power(c_safe, -gamma)
    
    # Terminal condition: at t=horizon-1, consume all resources
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Value function at terminal: V(m, z) = u(m) for all z
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))
    
    # Number of consumption scan points
    n_scan = max(budget, 300)
    
    # Precompute fracs for scan
    frac_min = 0.001
    fracs = np.linspace(frac_min, 1.0, n_scan)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
        
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        # Precompute V'_next for this iteration (needed for FOC refinement)
        V_prime_next = np.zeros((n_z, n_m))
        for zp in range(n_z):
            V_prime_next[zp, 0] = (V_next[zp, 1] - V_next[zp, 0]) / (m_grid[1] - m_grid[0])
            V_prime_next[zp, -1] = (V_next[zp, -1] - V_next[zp, -2]) / (m_grid[-1] - m_grid[-2])
            if n_m > 2:
                V_prime_next[zp, 1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (m_grid[2:] - m_grid[:-2])
        
        for z in range(n_z):
            trans_z = transition[z]
            
            # c_candidates[m_idx, scan_idx] = fracs[scan_idx] * m_grid[m_idx]
            c_candidates = fracs[np.newaxis, :] * m_grid[:, np.newaxis]  # shape (n_m, n_scan)
            
            # m_minus_c shape: (n_m, n_scan)
            m_minus_c = m_grid[:, np.newaxis] - c_candidates  # (n_m, n_scan)
            m_minus_c = np.maximum(m_minus_c, 0.0)
            
            # m_prime for each z': shape (n_z, n_m, n_scan)
            m_prime_all = R * m_minus_c[np.newaxis, :, :] + income[:, np.newaxis, np.newaxis]
            m_prime_all = np.clip(m_prime_all, m_min, m_max)
            
            # Compute continuation value for each z', m, c
            cont_all = np.zeros((n_z, n_m, n_scan))
            for zp in range(n_z):
                mp_flat = m_prime_all[zp].ravel()
                cont_flat = np.interp(mp_flat, m_grid, V_next[zp, :])
                cont_all[zp] = cont_flat.reshape(n_m, n_scan)
            
            # Weighted continuation: sum over z' of trans_z[z'] * cont_all[z']
            cont_weighted = np.tensordot(trans_z, cont_all, axes=(0, 0))  # (n_m, n_scan)
            
            # Utility of consumption: shape (n_m, n_scan)
            util_c = utility(c_candidates)
            
            # Total objective: (n_m, n_scan)
            obj_matrix = util_c + beta * cont_weighted
            
            # Find best scan index for each m
            best_idx = np.argmax(obj_matrix, axis=1)  # (n_m,)
            
            # Get best c from scan
            best_c_scan = c_candidates[np.arange(n_m), best_idx]
            
            # FOC-based refinement using bisection on the derivative
            refined_c = best_c_scan.copy()
            
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_init = best_c_scan[m_idx]
                
                # Define derivative function for bisection
                def deriv_at(c_val, m_val=m_val, trans_z=trans_z):
                    c_safe = np.maximum(c_val, 1e-12)
                    m_minus_c_val = np.maximum(m_val - c_safe, 0.0)
                    m_prime = R * m_minus_c_val + income  # shape (n_z,)
                    m_prime_clipped = np.clip(m_prime, m_min, m_max)
                    
                    # V'_next at m_prime
                    V_prime_at_mp = np.zeros(n_z)
                    for zp in range(n_z):
                        V_prime_at_mp[zp] = np.interp(m_prime_clipped[zp], m_grid, V_prime_next[zp, :])
                    
                    mu_c = marginal_utility(c_safe)
                    deriv = mu_c - beta * R * np.dot(trans_z, V_prime_at_mp)
                    return deriv
                
                # Bisection bounds: use neighborhood around best_c_scan
                c_lo = max(1e-12, c_init * 0.8 if c_init > 0.01 * m_val else 1e-12)
                c_hi = min(m_val, c_init * 1.2 if c_init < 0.99 * m_val else m_val)
                
                # Ensure c_lo < c_hi
                if c_lo >= c_hi:
                    c_lo = max(1e-12, m_val * 0.001)
                    c_hi = m_val
                
                # Check derivative at bounds
                deriv_lo = deriv_at(c_lo)
                deriv_hi = deriv_at(c_hi)
                
                if deriv_lo <= 0:
                    c_opt = c_lo
                elif deriv_hi >= 0:
                    c_opt = c_hi
                else:
                    # Derivative crosses zero between c_lo and c_hi
                    c_lo_b = c_lo
                    c_hi_b = c_hi
                    for _ in range(40):
                        c_mid = 0.5 * (c_lo_b + c_hi_b)
                        d_mid = deriv_at(c_mid)
                        if d_mid > 0:
                            c_lo_b = c_mid
                        else:
                            c_hi_b = c_mid
                        if abs(c_hi_b - c_lo_b) < 1e-14 * m_val:
                            break
                    c_opt = 0.5 * (c_lo_b + c_hi_b)
                
                # Clamp to valid range
                c_opt = np.clip(c_opt, 1e-12, m_val)
                refined_c[m_idx] = c_opt
            
            C_opt[z, :] = refined_c
            
            # Compute V_curr using the refined consumption (vectorized)
            c_safe = np.maximum(C_opt[z, :], 1e-12)
            m_minus_c = np.maximum(m_grid - c_safe, 0.0)
            m_prime = R * m_minus_c[np.newaxis, :] + income[:, np.newaxis]  # (n_z, n_m)
            m_prime_clipped = np.clip(m_prime, m_min, m_max)
            
            # Continuation values
            cont = np.zeros((n_z, n_m))
            for zp in range(n_z):
                cont[zp, :] = np.interp(m_prime_clipped[zp, :], m_grid, V_next[zp, :])
            
            cont_val = np.dot(trans_z, cont)  # (n_m,)
            V_curr[z, :] = utility(c_safe) + beta * cont_val
        
        consumption[t] = C_opt
        V_next = V_curr
    
    # Ensure terminal condition: c[5] = grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c_vals = consumption[t, z, :]
            c_vals = np.maximum(c_vals, 1e-12)
            c_vals = np.minimum(c_vals, m_grid + 1e-10)
            consumption[t, z, :] = c_vals
    
    return consumption
# EVOLVE-BLOCK-END
