"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    # Internal fine grid for backward induction
    fine_grid = np.geomspace(0.001, 40.0, 500)
    n_fine = len(fine_grid)
    
    # Utility and derivatives
    def crra(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)
            
    def inv_u_prime(v):
        # v = u'(c) -> c = (u')^{-1}(v)
        v = np.maximum(v, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / v
        else:
            return v ** (-1.0 / gamma)
            
    # V_prime[t, z, m] stores the derivative of the value function
    V_prime = np.zeros((horizon, n_z, n_fine))
    c_star = np.zeros((horizon, n_z, n_fine))
    
    # Terminal condition
    t_term = horizon - 1
    for z in range(n_z):
        V_prime[t_term, z, :] = u_prime(fine_grid)
        c_star[t_term, z, :] = fine_grid
        
    # Backward induction using Euler Equation Fixed Point
    for t in range(horizon - 2, -1, -1):
        # Create splines for V_prime[t+1, z, :] and its derivative
        splines_Vp = []
        splines_Vpp = []
        for z_prime in range(n_z):
            # V_prime is smooth, use CubicSpline
            sp = CubicSpline(fine_grid, V_prime[t+1, z_prime, :], bc_type='clamped')
            splines_Vp.append(sp)
            splines_Vpp.append(sp.derivative())
            
        for z in range(n_z):
            # Solve for V_prime[t, z, :]
            v_curr = np.zeros(n_fine)
            c_curr = np.zeros(n_fine)
            
            for i_m in range(n_fine):
                m = fine_grid[i_m]
                # Initial guess for v
                # If we consume everything, v = u'(m). If we save, v > u'(m).
                # Let's start with v = u'(m) * 1.1 or something.
                v = u_prime(m) * 1.05
                
                # Newton iterations
                for _ in range(50):
                    c = inv_u_prime(v)
                    c = np.clip(c, 1e-12, m)
                    
                    # Calculate h(v) and h'(v)
                    h_val = v
                    h_deriv = 1.0
                    u_pp_c = u_double_prime(c)
                    
                    # u_pp_c is negative
                    # term = R / u_pp_c is negative
                    # h'(v) = 1 - beta * R * sum P * S'(m') * d m' / d v
                    # d m' / d v = -R / u_pp_c
                    # h'(v) = 1 - beta * R * sum P * S'(m') * (-R / u_pp_c)
                    # h'(v) = 1 + beta * R^2 / u_pp_c * sum P * S'(m')
                    
                    for z_prime in range(n_z):
                        p = transition[z, z_prime]
                        if p == 0: continue
                        
                        m_next = R * (m - c) + income[z_prime]
                        # Clip m_next to grid for spline evaluation
                        m_next_clipped = np.clip(m_next, fine_grid[0], fine_grid[-1])
                        
                        v_p_next = splines_Vp[z_prime](m_next_clipped)
                        v_pp_next = splines_Vpp[z_prime](m_next_clipped)
                        
                        h_val -= beta * R * p * v_p_next
                        # Add to derivative
                        h_deriv += beta * (R * R) / u_pp_c * p * v_pp_next
                        
                    delta_v = -h_val / h_deriv
                    v_new = v + delta_v
                    v_new = np.maximum(v_new, u_prime(m)) # v must be >= u'(m) since c <= m
                    if np.abs(v_new - v) < 1e-8 * max(1.0, np.abs(v)):
                        v = v_new
                        break
                    v = v_new
                    
                    # Safety check
                    if not np.isfinite(v):
                        v = u_prime(m) * 1.05
                        break
                        
                v_curr[i_m] = v
                c_curr[i_m] = inv_u_prime(v)
                c_curr[i_m] = np.clip(c_curr[i_m], 1e-12, m)
                
            V_prime[t, z, :] = v_curr
            c_star[t, z, :] = c_curr
            
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
