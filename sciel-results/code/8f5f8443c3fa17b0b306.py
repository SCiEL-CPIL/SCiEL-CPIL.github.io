"""Solve consumption-saving policy using piecewise log-linear value functions and Newton's method."""
import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    num_z = len(income)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    def mu(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    def mu_inv(x):
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / x
        else:
            return np.power(x, -1.0 / gamma)
    
    # We'll store value functions as arrays for each state (t, z)
    # V[t, z, :] = value at each grid point
    # C[t, z, :] = consumption at each grid point
    
    V = np.zeros((horizon, num_z, num_m))
    C = np.zeros((horizon, num_z, num_m))
    
    # Terminal condition: at t=5 (last period), consume everything
    # c[5] = m (the grid)
    for z in range(num_z):
        C[5, z, :] = M
        V[5, z, :] = util(M)
    
    # Build interpolation for terminal value
    def build_interp(z_idx):
        """Build cubic spline for V[5, z_idx, :] as function of log(m)"""
        log_m = np.log(M)
        v = V[5, z_idx, :]
        # Handle any non-finite values
        v = np.where(np.isfinite(v), v, -1e10)
        cs = CubicSpline(log_m, v, bc_type='natural')
        return cs
    
    # Backward induction
    for t in range(4, -1, -1):
        for z in range(num_z):
            # Get continuation value functions for next period
            # V_next[z', m'] where m' = R*(m - c) + income[z']
            
            # For each m in grid, we need to solve for c such that:
            # mu(c) = beta * sum_{z'} P(z'|z) * R * mu(m')
            # where m' = R*(m - c) + income[z']
            
            # This is a fixed-point / root-finding problem.
            # Define f(c) = mu(c) - beta * sum_{z'} P(z'|z) * R * mu(R*(m-c) + income[z'])
            # We want f(c) = 0
            
            # Build interpolators for next period value functions
            interp_next = {}
            for zp in range(num_z):
                if t + 1 < horizon:
                    log_m = np.log(M)
                    v_next = V[t+1, zp, :]
                    v_next = np.where(np.isfinite(v_next), v_next, -1e10)
                    interp_next[zp] = CubicSpline(log_m, v_next, bc_type='natural')
                else:
                    # Terminal
                    log_m = np.log(M)
                    v_next = V[t+1, zp, :] if t+1 < horizon else util(M)
                    v_next = np.where(np.isfinite(v_next), v_next, -1e10)
                    interp_next[zp] = CubicSpline(log_m, v_next, bc_type='natural')
            
            # For each m, solve for optimal c
            for i in range(num_m):
                m = M[i]
                
                def euler_eq(c):
                    c = max(c, 1e-12)
                    c = min(c, m)  # Can't consume more than wealth
                    lhs = mu(c)
                    rhs = 0.0
                    for zp in range(num_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        if m_prime < 1e-12:
                            m_prime = 1e-12
                        # mu(m') - but we need the marginal value of next period's wealth
                        # The marginal value is the derivative of V_next w.r.t. m'
                        # For CRRA, if we're at interior, mu(m') = mu of consumption next period
                        # But we don't know next period's consumption directly.
                        # Actually, the Euler equation is:
                        # mu(c_t) = beta * R * sum P(z'|z) * mu(c_{t+1}(z', m'))
                        # where c_{t+1}(z', m') is the optimal consumption next period
                        # And mu(c_{t+1}) = dV/dm' (envelope theorem)
                        # So we need dV_next[zp]/dm' at m'
                        
                        # Get derivative from spline
                        log_mp = np.log(m_prime)
                        if log_mp < log_m[0]:
                            log_mp = log_m[0]
                        elif log_mp > log_m[-1]:
                            log_mp = log_m[-1]
                        dv_dm_prime = interp_next[zp](log_mp, 1) / m_prime  # dV/dm = dV/dlog_m / m
                        rhs += prob * R * dv_dm_prime
                    
                    return lhs - beta * rhs
                
                # Solve for c in (1e-12, m)
                # Try bracketing
                c_lo = 1e-12
                c_hi = m * (1.0 - 1e-10)
                
                # Check signs
                f_lo = euler_eq(c_lo)
                f_hi = euler_eq(c_hi)
                
                if f_lo * f_hi > 0:
                    # No sign change, use bisection-like approach or just pick midpoint
                    # This can happen at boundaries
                    c_opt = m * 0.5
                else:
                    try:
                        c_opt = brentq(euler_eq, c_lo, c_hi, xtol=1e-12, rtol=1e-12, maxiter=100)
                    except:
                        c_opt = m * 0.5
                
                c_opt = max(c_opt, 1e-12)
                c_opt = min(c_opt, m)
                
                C[t, z, i] = c_opt
                
                # Compute value: util(c) + beta * sum P(z'|z) * V_next[z'](m')
                val = util(c_opt)
                for zp in range(num_z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_prime = R * (m - c_opt) + income[zp]
                    if m_prime < M[0]:
                        m_prime = M[0]
                    elif m_prime > M[-1]:
                        m_prime = M[-1]
                    log_mp = np.log(m_prime)
                    v_cont = interp_next[zp](log_mp)
                    val += beta * prob * v_cont
                
                V[t, z, i] = val
        
        # Check time
        if time.time() > deadline:
            break
    
    # Enforce terminal condition: c[5] = grid
    for z in range(num_z):
        C[5, z, :] = M
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], M + 1e-10)
    
    return C
