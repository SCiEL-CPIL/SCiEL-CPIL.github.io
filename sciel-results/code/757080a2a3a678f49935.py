"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 138
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Dense internal grid for backward induction
    m_grid = np.geomspace(0.001, 40, 501)
    n_m = len(m_grid)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
    
    # Policy array: c[t, z, m_index]
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal: consume all
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
    
    # Value function on m_grid: V[t, z, m_index]
    V = np.zeros((horizon, n_z, n_m))
    for z in range(n_z):
        V[horizon - 1, z, :] = u(m_grid)
    
    # Precompute cubic splines for value functions at t+1
    # We'll update these as we go backward
    V_splines = [None] * horizon  # V_splines[t] is spline for V[t]
    
    # Initialize terminal splines
    for z in range(n_z):
        # Use a log-spaced grid for better spline behavior
        V_splines[horizon - 1] = V_splines[horizon - 1] if V_splines[horizon - 1] is not None else {}
        if V_splines[horizon - 1] is None:
            V_splines[horizon - 1] = {}
        V_splines[horizon - 1][z] = CubicSpline(m_grid, V[horizon - 1, z, :], bc_type='natural')
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 2:
            break
        
        for z in range(n_z):
            # Get splines for next period
            next_splines = V_splines[t + 1]
            
            for i in range(n_m):
                m_i = m_grid[i]
                
                if m_i < 1e-10:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                def euler_residual(c):
                    if c <= 0 or c >= m_i - 1e-12:
                        return 1e10
                    m_save = R * (m_i - c)
                    total_v_prime = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_save + income[zp]
                        m_zp_clipped = np.clip(m_zp, 1e-12, 40.0)
                        # Use spline derivative - much faster and smoother
                        v_prime_val = next_splines[zp](m_zp_clipped, 1)
                        total_v_prime += prob * v_prime_val
                    return u_prime(c) - beta * R * total_v_prime
                
                # Minimize absolute value of Euler residual
                c_lo = 1e-10
                c_hi = m_i * (1 - 1e-10)
                if c_hi <= c_lo:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                try:
                    res = minimize_scalar(lambda c: abs(euler_residual(c)), bounds=(c_lo, c_hi), 
                                         method='bounded', options={'xatol': 1e-9, 'maxiter': 60})
                    c_opt = res.x
                except Exception:
                    c_opt = m_i * 0.5
                
                c_opt = np.clip(c_opt, c_lo, c_hi)
                c_policy[t, z, i] = c_opt
                
                # Compute V[t, z, i]
                m_save = R * (m_i - c_opt)
                total = 0.0
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_zp = m_save + income[zp]
                    m_zp_clipped = np.clip(m_zp, 1e-12, 40.0)
                    v_val = next_splines[zp](m_zp_clipped, 0)
                    total += prob * v_val
                V[t, z, i] = u(c_opt) + beta * total
        
        # Build splines for current period after computing all z
        V_splines[t] = {}
        for z in range(n_z):
            V_splines[t][z] = CubicSpline(m_grid, V[t, z, :], bc_type='natural')
    
    # Interpolate to resource_grid using cubic spline for smoothness
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            # Use cubic spline interpolation for the policy
            try:
                c_spline = CubicSpline(m_grid, c_policy[t, z, :], bc_type='natural')
                result[t, z, :] = c_spline(resource_grid)
            except Exception:
                result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
                result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
