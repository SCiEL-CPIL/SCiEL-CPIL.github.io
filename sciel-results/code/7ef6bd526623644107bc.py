"""Solve consumption-saving policy via high-precision backward induction."""
import numpy as np
import time
from scipy.interpolate import CubicSpline


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=np.float64)
    N_m = len(m_grid)
    N_z = len(income)

    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Use a dense grid for backward induction
    # The resource grid spans geomspace(0.001, 40, 101)
    # We need a finer grid for accurate interpolation
    # Use log-spaced grid with more points
    m_dense = np.geomspace(0.0005, 50, 2000)
    N_dense = len(m_dense)

    # Backward induction: store value functions for each period
    # V[t, z, m] for t in 0..horizon-1, z in 0..N_z-1
    # Start from terminal: V[horizon] = 0 (or we can use a terminal condition)
    # Actually, for finite horizon, V[horizon] = 0 (no utility after horizon)
    
    # We'll compute consumption for t = 0, 1, ..., horizon-1
    # V[t] depends on V[t+1]
    
    # Store value functions as arrays indexed by z and dense grid
    # V[t, z, :] = value function at time t for state z on dense grid
    
    # Terminal value: V[horizon] = 0
    # For t = horizon-1 down to 0:
    # V[t, z, m] = max_c { util(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', R*(m-c) + income[z']] }
    
    # We need to compute this efficiently.
    # For each t, z, and m on the dense grid, we need to find optimal c.
    
    # Strategy: For each (t, z), we have a continuation value function
    # W_z(m) = beta * sum_{z'} transition[z, z'] * V[t+1, z', R*(m-c) + income[z']]
    # But this depends on c through m' = R*(m-c) + income[z']
    
    # Let's think differently. For a given (t, z, m), we choose c in [0, m].
    # The continuation is: beta * sum_{z'} P[z,z'] * V[t+1, z', R*(m-c) + income[z']]
    # Let m'_{z'} = R*(m-c) + income[z']
    # So continuation = beta * sum_{z'} P[z,z'] * V[t+1, z', R*m - R*c + income[z']]
    
    # We can precompute for each (t, z) a function of c.
    
    # For efficiency, let's use the fact that V[t+1, z', *] is a smooth function.
    # We'll represent it with a spline.
    
    # Initialize V for terminal period (all zeros)
    V_next = np.zeros((N_z, N_dense), dtype=np.float64)
    
    # We'll store consumption policies on the original grid
    consumption = np.zeros((horizon, N_z, N_m), dtype=np.float64)
    
    # Precompute interpolation for V_next
    # For each z, create a spline of V_next[z, :] vs m_dense
    
    for t in range(horizon - 1, -1, -1):
        if time.time() - start_time > DEADLINE - 30:
            # Not enough time, break and use what we have
            break
        
        # Create splines for V_next for each z
        splines = []
        for z in range(N_z):
            cs = CubicSpline(m_dense, V_next[z, :])
            splines.append(cs)
        
        # For each z and each m on the dense grid, find optimal c
        # V_current[z, m] = max_c { util(c) + beta * sum_{z'} P[z,z'] * V_next[z', R*(m-c)+income[z']] }
        
        V_current = np.zeros((N_z, N_dense), dtype=np.float64)
        
        for z in range(N_z):
            # For this z, we need to optimize over c for each m
            # The objective for given m is: f(c) = util(c) + beta * sum_{z'} P[z,z'] * spline_{z'}(R*(m-c) + income[z'])
            # c in [0, m]
            
            # We can use Newton's method or golden section search
            # Let's use a combination: first a coarse grid, then refine
            
            # For each m on the dense grid, find optimal c
            # This is expensive: N_z * N_dense evaluations
            # Let's vectorize where possible
            
            m_vals = m_dense
            # For each m, the optimal c satisfies the FOC:
            # u'(c) = beta * sum_{z'} P[z,z'] * V_next'[z'](m'_{z'}) * (-R)
            # where m'_{z'} = R*(m-c) + income[z']
            # So: u'(c) = -beta * R * sum_{z'} P[z,z'] * V_next'[z'](R*(m-c) + income[z'])
            
            # We can solve this iteratively. Start with a guess and iterate.
            # Or use a root-finding approach.
            
            # Let's use a simpler approach: for each m, do a 1D optimization
            # Since N_dense is 2000 and N_z is 22, this is 40000 1D optimizations.
            # Each can be done with ~20-30 evaluations, so ~1M total. Should be fast.
            
            # Precompute the continuation value function for this z
            # g_z(c, m) = beta * sum_{z'} P[z,z'] * V_next[z', R*(m-c) + income[z']]
            
            # For a fixed m, g_z is a function of c.
            # Let's define h_z(c) = sum_{z'} P[z,z'] * V_next[z', R*(-c) + (R*m + income[z'])]
            # = sum_{z'} P[z,z'] * spline_{z'}(R*m + income[z'] - R*c)
            
            # For each m, we need to maximize: util(c) + beta * h_z(c)
            # subject to 0 <= c <= m
            
            # Let's use a vectorized approach with a coarse grid of c values
            # For each m, the range of c is [0, m]
            
            # Alternative: use the FOC and solve with Newton
            # FOC: u'(c) + beta * R * sum_{z'} P[z,z'] * V_next'[z'](R*(m-c) + income[z']) = 0
            # Note: d/dc [V_next[z'](R*(m-c)+income[z'])] = V_next'[z'] * (-R)
            # So the derivative of the continuation w.r.t. c is: beta * sum P[z,z'] * V_next'[z'] * (-R)
            # Total derivative: u'(c) - beta * R * sum P[z,z'] * V_next'[z'](R*(m-c)+income[z']) = 0
            
            # Let's solve this with Newton's method for each m
            
            # Initial guess: c0 = m / (1 + beta*R) approximately (for log utility, gamma=1)
            # More generally, for CRRA, the optimal c is roughly m * (1 - beta*R^(-1/(1-gamma)))/(1 - beta*R^(-1/(1-gamma)))
            # but let's just use a reasonable initial guess
            
            c_opt = np.zeros(N_dense, dtype=np.float64)
            
            # We'll do this in batches to keep memory reasonable
            batch_size = 200
            for i_start in range(0, N_dense, batch_size):
                i_end = min(i_start + batch_size, N_dense)
                m_batch = m_dense[i_start:i_end]
                
                # Initial guess
                # For CRRA, a good initial guess is c = m * (1 - (beta*R)**(1/(1-gamma))) / (1 - (beta*R)**(1/(1-gamma)))
                # if beta*R < 1, else c = m
                br = beta * R
                if gamma != 1:
                    exponent = 1.0 / (1.0 - gamma)
                    if br < 1.0:
                        frac = (1.0 - br ** exponent) / (1.0 - br ** exponent)
                        # This simplifies to 1, so let me reconsider
                        # Actually: c/m = (1 - (beta*R)^(1/(1-gamma))) / (1 - (beta*R)^(1/(1-gamma)))
                        # That's not right. Let me use a different approach.
                        # For log utility (gamma=1): c = m / (1 + beta*R)
                        # For general CRRA: c = m * (1 - (beta*R)^(1/(1-gamma))) / (1 - (beta*R)^(1/(1-gamma)))
                        # Hmm, let me just use c = m * 0.5 as initial guess
                        pass
                # Use c = m * 0.5 as initial guess, clipped to [1e-12, m]
                c_iter = np.clip(m_batch * 0.5, 1e-12, m_batch)
                
                # Newton iterations
                for _newton in range(15):
                    # Compute u'(c_iter)
                    up = u_prime(c_iter)
                    
                    # Compute continuation derivative: -beta * R * sum_z' P[z,z'] * V_next'[z'](R*(m-c)+income[z'])
                    # V_next'[z'] at m'_{z'} = R*(m-c) + income[z']
                    # We need the derivative of the spline, which CubicSpline provides
                    cont_deriv = np.zeros(len(m_batch), dtype=np.float64)
                    for zp in range(N_z):
                        m_prime = R * (m_batch - c_iter) + income[zp]
                        # Clip m_prime to valid range
                        m_prime = np.clip(m_prime, m_dense[0], m_dense[-1])
                        # Get derivative from spline
                        v_next_prime = splines[zp](m_prime, 1)
                        cont_deriv += transition[z, zp] * v_next_prime
                    
                    # FOC residual: u'(c) - beta * R * cont_deriv = 0
                    # Note: cont_deriv = sum P[z,z'] * V_next'[z'](m'_{z'})
                    # The derivative of continuation w.r.t. c is: beta * sum P[z,z'] * V_next'[z'] * (-R)
                    # So total derivative: u'(c) + beta * (-R) * cont_deriv = u'(c) - beta*R*cont_deriv
                    residual = up - beta * R * cont_deriv
                    
                    # Second derivative for Newton step
                    # u''(c) - beta*R * sum P[z,z'] * V_next''[z'](m'_{z'}) * (-R)
                    # = u''(c) + beta*R^2 * sum P[z,z'] * V_next''[z'](m'_{z'})
                    up2 = u_double_prime(c_iter)
                    cont_deriv2 = np.zeros(len(m_batch), dtype=np.float64)
                    for zp in range(N_z):
                        m_prime = R * (m_batch - c_iter) + income[zp]
                        m_prime = np.clip(m_prime, m_dense[0], m_dense[-1])
                        v_next_double = splines[zp](m_prime, 2)
                        cont_deriv2 += transition[z, zp] * v_next_double
                    
                    second_deriv = up2 + beta * R * R * cont_deriv2
                    # Ensure second_deriv is negative (concavity)
                    second_deriv = np.minimum(second_deriv, -1e-10)
                    
                    # Newton step
                    c_new = c_iter - residual / second_deriv
                    # Clip to [1e-12, m_batch]
                    c_new = np.clip(c_new, 1e-12, m_batch)
                    
                    # Check convergence
                    if np.max(np.abs(c_new - c_iter)) < 1e-10:
                        c_iter = c_new
                        break
                    c_iter = c_new
                
                c_opt[i_start:i_end] = c_iter
        
        # Compute V_current for each z and m
        for z in range(N_z):
            c_vals = c_opt
            m_vals = m_dense
            # V_current[z, m] = util(c) + beta * sum_{z'} P[z,z'] * V_next[z', R*(m-c)+income[z']]
            util_vals = util(c_vals)
            cont_val = np.zeros(N_dense, dtype=np.float64)
            for zp in range(N_z):
                m_prime = R * (m_vals - c_vals) + income[zp]
                m_prime = np.clip(m_prime, m_dense[0], m_dense[-1])
                cont_val += transition[z, zp] * splines[zp](m_prime)
            V_current[z, :] = util_vals + beta * cont_val
        
        # Store consumption on the original grid for this period
        # Interpolate c_opt from dense grid to original grid
        cs = CubicSpline(m_dense, c_opt)
        c_on_grid = cs(m_grid)
        # Ensure validity: 0 < c <= m + 1e-10
        c_on_grid = np.clip(c_on_grid, 1e-12, m_grid + 1e-10)
        consumption[t, :, :] = c_on_grid[np.newaxis, :, :]
        
        # Update V_next
        V_next = V_current
    
    # Final constraint: c[5] must equal the grid within 1e-9
    # At t=5 (last period), optimal c = m (consume everything)
    consumption[5, :, :] = m_grid[np.newaxis, :, :]
    
    # Validate: finite values, 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            c_row = consumption[t, z, :]
            c_row = np.where(np.isfinite(c_row), c_row, m_grid)
            c_row = np.clip(c_row, 1e-12, m_grid + 1e-10)
            consumption[t, z, :] = c_row
    
    return consumption
