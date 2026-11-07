import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M_out = len(resource_grid)
    Z = len(income)

    # Use a fine internal grid for high accuracy
    # 501 points for better resolution
    M_int = 501
    m_grid = np.geomspace(0.001, 40, M_int)
    
    # Precompute grid differences for slope calculation
    dm = np.diff(m_grid)

    # consumption[t, z, m_index] on the output grid
    consumption = np.zeros((horizon, Z, M_out))

    # Helper for marginal utility
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return 1.0 / c
        else:
            return c ** (-g)

    # Helper for log marginal utility
    def log_u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return -np.log(c)
        else:
            return -g * np.log(c)

    # Terminal condition: at t=5 (last decision), consume everything.
    # c_5(m, z) = m
    c_next = np.zeros((Z, M_int))
    for z in range(Z):
        c_next[z] = m_grid

    # Build cubic spline for terminal condition
    c_splines = [CubicSpline(m_grid, c_next[z]) for z in range(Z)]

    # Set terminal period consumption on output grid
    for z in range(Z):
        consumption[horizon - 1, z] = resource_grid

    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule (consume all)
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break

        c_curr = np.zeros((Z, M_int))
        
        # Precompute slopes of c_next for derivative calculations using spline
        # slopes[z, i] = dc_next/dm at m_grid[i]
        slopes = np.zeros((Z, M_int))
        for z in range(Z):
            slopes[z] = c_splines[z](m_grid, 1)

        for z in range(Z):
            # We solve for s in [0, 1) such that F(s) = 0
            # F(s) = log(u'(m(1-s))) - log(beta * R * sum_{z'} P(z'|z) u'(c_next(m', z')))
            
            # Initial guess
            s = np.full(M_int, 0.2)
            # Bisection bounds
            s_lo = np.zeros(M_int)
            s_hi = np.full(M_int, 0.999999)
            
            # We will use a hybrid Newton-Bisection method
            # For each m, we maintain lo/hi.
            
            for _ in range(25):
                if time.time() > deadline:
                    break
                
                c = m_grid * (1.0 - s)
                c = np.maximum(c, 1e-12)
                
                # LHS
                lhs = log_u_prime(c, gamma)
                
                # RHS components
                # m' = R * m * s + income[z']
                m_prime = R * m_grid[:, None] * s[:, None] + income[None, :]
                
                # Clip m_prime to grid bounds for spline evaluation
                m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                
                u_prime_sum = np.zeros(M_int)
                u_prime_prime_sum = np.zeros(M_int) # For V'(s)
                
                for zp in range(Z):
                    # Evaluate spline for c_next and its derivative
                    c_next_val = c_splines[zp](m_prime_clipped[:, zp])
                    c_next_val = np.maximum(c_next_val, 1e-12)
                    
                    # u'(c_next)
                    u_val = u_prime(c_next_val, gamma)
                    
                    # u''(c_next) is needed for V'(s)
                    # u'(c) = c^-gamma
                    # u''(c) = -gamma c^(-gamma-1)
                    if gamma == 1:
                        u_prime_prime_val = -1.0 / (c_next_val * c_next_val)
                    else:
                        u_prime_prime_val = -gamma * (c_next_val ** (-gamma - 1))
                    
                    # Slope of c_next at m' from spline derivative
                    slope_val = c_splines[zp](m_prime_clipped[:, zp], 1)
                    
                    # dm'/ds = R * m
                    dm_ds = R * m_grid
                    
                    # Contribution to V(s)
                    u_prime_sum += transition[z, zp] * u_val
                    
                    # Contribution to V'(s)
                    # V'(s) = sum P * u''(c_next) * dc_next/dm' * dm'/ds
                    u_prime_prime_sum += transition[z, zp] * u_prime_prime_val * slope_val * dm_ds
                
                rhs_val = beta * R * u_prime_sum
                rhs_val = np.maximum(rhs_val, 1e-12)
                rhs = np.log(rhs_val)
                
                F = lhs - rhs
                
                # Derivatives
                # F'(s) = LHS' - RHS'
                # LHS' = gamma / (1 - s)
                d_lhs = gamma / (1.0 - s)
                
                # RHS' = d/ds log(beta R V(s)) = V'(s) / V(s)
                # V(s) = u_prime_sum
                # V'(s) = u_prime_prime_sum
                d_rhs = u_prime_prime_sum / u_prime_sum
                
                dF = d_lhs - d_rhs
                
                # Newton step
                # s_new = s - F / dF
                # Handle division by zero
                dF_safe = np.where(np.abs(dF) < 1e-10, 1e-10, dF)
                s_new = s - F / dF_safe
                
                # Bisection update
                mask_pos = F > 0
                mask_neg = F < 0
                
                s_hi[mask_pos] = s[mask_pos]
                s_lo[mask_neg] = s[mask_neg]
                
                # Clip Newton step to bisection bounds
                s_new = np.clip(s_new, s_lo, s_hi)
                
                max_diff = np.max(np.abs(s_new - s))
                
                s = s_new
                
                if max_diff < 1e-8:
                    break
            
            # Final c
            c_curr[z] = m_grid * (1.0 - s)
            c_curr[z] = np.maximum(c_curr[z], 1e-12)
            c_curr[z] = np.minimum(c_curr[z], m_grid + 1e-10)

        # Store the policy for this period on the output grid using spline interpolation
        for z in range(Z):
            c_curr_spline = CubicSpline(m_grid, c_curr[z])
            consumption[t, z] = c_curr_spline(resource_grid)

        # Update c_next and splines
        c_next = c_curr.copy()
        c_splines = [CubicSpline(m_grid, c_next[z]) for z in range(Z)]

    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c

    # Enforce terminal condition exactly
    for z in range(Z):
        consumption[horizon - 1, z] = resource_grid

    return consumption
