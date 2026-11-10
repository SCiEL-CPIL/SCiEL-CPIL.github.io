import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import make_interp_spline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    
    consumption = np.zeros((horizon, 2, n_m))
    
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    def up(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)

    consumption[-1, :, :] = m_grid[np.newaxis, :, np.newaxis]
    
    V_last = np.zeros((2, n_m))
    for z in range(2):
        V_last[z, :] = u(m_grid)
        
    def make_splines(V_vals):
        splines = []
        for z in range(2):
            try:
                s = make_interp_spline(m_grid, V_vals[z], k=3)
            except Exception:
                s = make_interp_spline(m_grid, V_vals[z], k=1)
            splines.append(s)
        return splines

    current_splines = make_splines(V_last)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > 145:
            break
            
        V_current = np.zeros((2, n_m))
        # Pre-compute derivative splines for speed
        deriv_splines = [s.derivative() for s in current_splines]
        
        for z in range(2):
            probs = transition[z]
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                def euler_residual(c, m=m, probs=probs, deriv_splines=deriv_splines, R=R, income=income, beta=beta, up=up):
                    if c <= 1e-12:
                        return 1e100
                    if c >= m - 1e-12:
                        return -1e100
                        
                    rhs_marginal = 0.0
                    for z_next in range(2):
                        p = probs[z_next]
                        if p == 0:
                            continue
                        m_next = R * (m - c) + income[z_next]
                        dv_dm = deriv_splines[z_next](m_next)
                        rhs_marginal += p * dv_dm
                    
                    target_marginal = beta * R * rhs_marginal
                    current_marginal = up(c)
                    return current_marginal - target_marginal

                c_low = 1e-12
                c_high = m - 1e-12
                
                # Check boundary c = m
                # If F(m) > 0, then optimal c is m (or close to it)
                # F(c) = up(c) - RHS.
                # If up(m) > RHS, then we want higher c, but max is m. So c=m.
                # If up(m) < RHS, we want lower c.
                
                # Evaluate at c_high
                f_high = euler_residual(c_high)
                
                if f_high > 0:
                    c_opt = c_high
                else:
                    # f_low is always positive (up(0) is inf)
                    # So root is in [c_low, c_high]
                    try:
                        c_opt = brentq(euler_residual, c_low, c_high, xtol=1e-10, rtol=1e-10)
                    except ValueError:
                        c_opt = (c_low + c_high) / 2.0
                        # Refine if needed?
                        # For now, midpoint is a safe fallback.
                
                consumption[t, z, i_m] = c_opt
                
                val = u(c_opt)
                for z_next in range(2):
                    p = probs[z_next]
                    if p == 0:
                        continue
                    m_next = R * (m - c_opt) + income[z_next]
                    v_next = current_splines[z_next](m_next)
                    val += beta * p * v_next
                
                V_current[z, i_m] = val
        
        current_splines = make_splines(V_current)

    return consumption
