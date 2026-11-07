def solve_consumption(model, resource_grid, budget):
    import numpy as np
    import time
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']
    
    T = len(resource_grid)
    Z = len(income)
    
    c = np.zeros((H, Z, T))
    c[H-1, :, :] = resource_grid[None, :]
    
    start_time = time.time()
    deadline = 140.0
    
    # Use a finer internal grid for computation, then interpolate back
    # The resource_grid has 101 points. Let's use a denser grid for the Bellman equation.
    # However, the output must be on the original resource_grid.
    # We can solve on a denser grid and then interpolate the policy back.
    # Or we can just use the original grid but with better methods.
    # Given the time limit, let's try to solve on the original grid but with vectorized operations.
    
    # Actually, the key is accuracy. Let's use the original grid.
    
    for t in range(H-2, -1, -1):
        if time.time() > deadline:
            break
        
        for z in range(Z):
            # Use a good initial guess
            # If t+1 is the last period, c_next = m.
            # So c_t(m) = m - (1/R) * (1/beta) * E[1/c_next(m')]^{1/gamma} ... no, that's not right.
            # Let's just start with 0.5*m or interpolate from t+1.
            
            # Initial guess: interpolate c[t+1] at R*(m - 0.5*m) + income?
            # Or just 0.5*m.
            c_curr = 0.5 * resource_grid
            
            # Newton's method
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                # Vectorized computation
                lhs = c_curr ** (-gamma)
                rhs = np.zeros_like(c_curr)
                f_prime_rhs = np.zeros_like(c_curr)
                
                for zp in range(Z):
                    p = transition[z, zp]
                    if p == 0:
                        continue
                    
                    mp = R * (resource_grid - c_curr) + income[zp]
                    
                    # Interpolate c[t+1, zp]
                    c_next_vals = np.interp(mp, resource_grid, c[t+1, zp, :])
                    c_next_vals = np.maximum(c_next_vals, 1e-12)
                    
                    rhs += p * (c_next_vals ** (-gamma))
                    
                    # Derivative
                    delta = 1e-4
                    mp_plus = mp + delta
                    mp_minus = mp - delta
                    c_next_plus = np.interp(mp_plus, resource_grid, c[t+1, zp, :])
                    c_next_minus = np.interp(mp_minus, resource_grid, c[t+1, zp, :])
                    c_next_plus = np.maximum(c_next_plus, 1e-12)
                    c_next_minus = np.maximum(c_next_minus, 1e-12)
                    
                    dc_dmp = (c_next_plus - c_next_minus) / (2 * delta)
                    
                    deriv_term = gamma * R * (c_next_vals ** (-gamma - 1)) * dc_dmp
                    f_prime_rhs += p * deriv_term
                
                rhs *= beta * R
                f_prime_rhs *= beta * R
                
                lhs_prime = -gamma * (c_curr ** (-gamma - 1))
                
                F = lhs - rhs
                F_prime = lhs_prime - f_prime_rhs
                
                # Newton step
                mask = np.abs(F_prime) > 1e-12
                c_update = np.zeros_like(c_curr)
                c_update[mask] = F[mask] / F_prime[mask]
                
                c_new = c_curr - c_update
                c_new = np.clip(c_new, 1e-12, resource_grid)
                
                if np.max(np.abs(c_new - c_curr)) < 1e-10:
                    c_curr = c_new
                    break
                c_curr = c_new
            
            c[t, z, :] = c_curr
    
    # Ensure final period is exact
    c[H-1, :, :] = resource_grid[None, :]
    
    # Final safety check: clip to valid range
    c = np.clip(c, 1e-12, resource_grid[None, None, :] + 1e-10)
    # Ensure c[5] (last period) is exactly the grid
    c[H-1, :, :] = resource_grid[None, :]
    
    return c
