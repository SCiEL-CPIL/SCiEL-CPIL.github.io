import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = 2
    
    # Derivative of utility
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # c_opt[t, z, m]
    c_opt = np.zeros((horizon, n_z, n_m))
    
    # Last period: consume all
    for z in range(n_z):
        c_opt[horizon - 1, z, :] = resource_grid
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # We solve for c[m] for each m in resource_grid
            c_policy = np.zeros(n_m)
            
            # Pre-compute interpolation parameters for next period's policy
            # We need to interpolate c_opt[t+1, zp, :]
            # np.interp works on 1D arrays.
            
            for i_m in range(n_m):
                m = resource_grid[i_m]
                
                # Define the Euler equation residual
                def euler_residual(c):
                    if c <= 1e-12:
                        return 1e10 # Push away from 0
                    if c >= m:
                        return -1e10 # Push away from m (if m is the bound)
                        
                    # LHS: u'(c)
                    lhs = util_prime(c)
                    
                    # RHS: beta * R * E[u'(c_next)]
                    rhs = 0.0
                    for zp in range(n_z):
                        # Next state resources
                        m_next = R * (m - c) + income[zp]
                        
                        # Clip m_next to grid range for interpolation
                        # Note: if m_next is outside, we might extrapolate or clip.
                        # The grid is 0.001 to 40.
                        # If m < c, m_next could be negative? No, c <= m is enforced by brentq bounds.
                        # But if c is close to m, m_next is small.
                        # If m_next < resource_grid[0], we clip to resource_grid[0] for interpolation?
                        # Or should we extrapolate? 
                        # The grid starts at 0.001. If m_next is very small, c_next will be very small.
                        # Let's clip to [resource_grid[0], resource_grid[-1]] for safety in interp.
                        m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                        
                        # Interpolate c_next
                        c_next = np.interp(m_next_clipped, resource_grid, c_opt[t+1, zp, :])
                        
                        # Add to expected RHS
                        rhs += transition[z, zp] * util_prime(c_next)
                    
                    rhs *= beta * R
                    return lhs - rhs
                
                # Bounds for brentq
                # c must be in (0, m]
                # We can use a small epsilon for lower bound
                c_low = 1e-6
                c_high = m * 0.999999
                
                # Check signs at bounds to ensure a root exists
                # If not, the optimum might be at the boundary
                f_low = euler_residual(c_low)
                f_high = euler_residual(c_high)
                
                if f_low * f_high > 0:
                    # No sign change, check which boundary is better
                    # This can happen if the solution is at the boundary
                    # For CRRA, usually interior.
                    # If f_low > 0 and f_high > 0, u'(c) is always > RHS, so increase c?
                    # u'(c) decreases with c.
                    # If LHS > RHS at low c, and LHS > RHS at high c, then we should increase c?
                    # Wait, if LHS > RHS, we are consuming too little (marginal utility of current is higher than future).
                    # So we should increase c.
                    # If f_low > 0 and f_high > 0, the root is to the right of c_high?
                    # But c_high is almost m.
                    # If f_low < 0 and f_high < 0, the root is to the left of c_low?
                    # Unlikely for c_low = 1e-6.
                    
                    # Let's just pick the bound with the smaller absolute residual or check objective?
                    # Actually, let's just try to find the root in a wider range or handle boundary.
                    # For robustness, if no sign change, we can fall back to a grid search for this specific point?
                    # Or just pick the midpoint?
                    # Let's assume interior solution for now and handle edge cases if they cause issues.
                    # For now, if no sign change, use c_high (consume almost all) if f_high < 0?
                    # If f_high < 0, LHS < RHS, so u'(c) is too low, need to decrease c.
                    # If f_high > 0, LHS > RHS, need to increase c (but can't).
                    
                    # Let's try a smaller c_high if f_high > 0? No, c_high is max.
                    # If f_low > 0 and f_high > 0, we are always "saving too much" relative to the Euler eq?
                    # This implies we should consume more. But we are at max consumption.
                    # So c_opt = m.
                    # If f_low < 0 and f_high < 0, we are always "consuming too much".
                    # So c_opt = 0 (or min).
                    
                    if f_high > 0:
                        c_policy[i_m] = m
                    else:
                        c_policy[i_m] = c_low
                else:
                    try:
                        c_policy[i_m] = brentq(euler_residual, c_low, c_high, xtol=1e-10)
                    except ValueError:
                        # Fallback if brentq fails
                        c_policy[i_m] = (c_low + c_high) / 2.0
                        
            c_opt[t, z, :] = c_policy
            
    return c_opt
