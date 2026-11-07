"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    def crra(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    # Fine internal grid for backward induction
    # Increased resolution for better accuracy
    fine_grid = np.geomspace(0.001, 40.0, 800)
    n_fine = len(fine_grid)
    
    # c fractions for optimization
    # Increased resolution to better approximate the optimal consumption
    c_frac = np.linspace(0.001, 1.0, 601)
    n_c = len(c_frac)
    
    # V[t, z, m_fine]
    V = np.zeros((horizon, n_z, n_fine))
    c_star = np.zeros((horizon, n_z, n_fine))
    
    # Terminal condition
    t_term = horizon - 1
    for z in range(n_z):
        V[t_term, z, :] = crra(fine_grid)
        c_star[t_term, z, :] = fine_grid
    
    # Precompute for vectorization
    # For each m_fine, c_vals = m * c_frac
    # m_minus_c = m - c_vals -> (n_fine, n_c)
    m_vals = fine_grid[:, np.newaxis]  # (n_fine, 1)
    c_frac_exp = c_frac[np.newaxis, :]  # (1, n_c)
    c_vals = m_vals * c_frac_exp  # (n_fine, n_c)
    c_vals = np.maximum(c_vals, 1e-12)
    c_vals = np.minimum(c_vals, m_vals + 1e-10)
    
    m_minus_c = m_vals - c_vals  # (n_fine, n_c)
    
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # m_prime = R * (m - c) + income[z']
            # shape: (n_fine, n_c, n_z)
            m_prime = R * m_minus_c[:, :, np.newaxis] + income[np.newaxis, np.newaxis, :]
            
            # Continuation value: for each (i_m, i_c, z'), interpolate V[t+1, z', :] at m_prime
            # We'll compute this as (n_fine, n_c, n_z)
            cont_vals = np.zeros((n_fine, n_c, n_z))
            for z_prime in range(n_z):
                # m_prime[:, :, z_prime] has shape (n_fine, n_c)
                # Interpolate along fine_grid
                # np.interp doesn't vectorize over the first axis easily, so we do it per z_prime
                # Actually, we can use np.interp with the flattened array or do it row by row
                # Better: use scipy.interpolate or do manual interpolation
                # For speed, let's do it vectorized using searchsorted
                mp_flat = m_prime[:, :, z_prime].ravel()  # (n_fine * n_c,)
                idx = np.searchsorted(fine_grid, mp_flat, side='right') - 1
                idx = np.clip(idx, 0, n_fine - 2)
                x0 = fine_grid[idx]
                x1 = fine_grid[idx + 1]
                y0 = V[t + 1, z_prime, idx]
                y1 = V[t + 1, z_prime, idx + 1]
                # Linear interpolation
                frac = (mp_flat - x0) / (x1 - x0)
                frac = np.clip(frac, 0, 1)
                interp_vals = y0 + frac * (y1 - y0)
                cont_vals[:, :, z_prime] = interp_vals.reshape(n_fine, n_c)
            
            # Expected continuation: sum over z' of transition[z, z'] * cont_vals
            # (n_fine, n_c)
            expected_cont = np.sum(transition[z, np.newaxis, np.newaxis, :] * cont_vals, axis=2)
            
            # Objective
            obj = crra(c_vals) + beta * expected_cont  # (n_fine, n_c)
            
            idx_best = np.argmax(obj, axis=1)  # (n_fine,)
            c_star[t, z, :] = c_vals[np.arange(n_fine), idx_best]
            V[t, z, :] = obj[np.arange(n_fine), idx_best]
    
    # Interpolate c_star onto original resource_grid
    consumption = np.zeros((horizon, n_z, n_grid))
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, c_star[t, z, :])
    
    # Terminal condition
    consumption[t_term, :, :] = resource_grid[np.newaxis, :]
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
    
    assert np.all(np.isfinite(consumption)), "Non-finite values"
    
    return consumption
# EVOLVE-BLOCK-END
