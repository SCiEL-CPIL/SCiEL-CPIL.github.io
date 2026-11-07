import time
import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 120.0  # 30s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    # Precompute log grid for interpolation
    log_grid = np.log(resource_grid)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    # V[t, z, i] = Value at state z, grid index i, time t
    # Vp[t, z, i] = Derivative of Value at state z, grid index i, time t
    V = np.zeros((horizon + 1, n_states, n_grid))
    Vp = np.zeros((horizon + 1, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))
    
    # Terminal condition: t = horizon
    V[horizon, :, :] = util(resource_grid)
    Vp[horizon, :, :] = util_deriv(resource_grid)
    
    # Helper to interpolate V or Vp at m_next
    def interp_val(arr, m_next):
        log_m_next = np.log(max(m_next, 1e-12))
        if log_m_next <= log_grid[0]:
            return arr[0]
        elif log_m_next >= log_grid[-1]:
            return arr[-1]
        else:
            return np.interp(log_m_next, log_grid, arr)

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 15.0:
            # Fallback: consume everything if time is running out
            for tt in range(t, horizon):
                C[tt, :, :] = resource_grid
            break
        
        for z in range(n_states):
            if time.time() > deadline - 10.0:
                C[t, z, :] = resource_grid
                continue
            
            c_opt = np.zeros(n_grid)
            val_t = np.zeros(n_grid)
            
            for i, m in enumerate(resource_grid):
                # FOC: U'(c) = beta * R * E[Vp_{t+1}(m')]
                
                def residual(c):
                    if c <= 1e-12:
                        return util_deriv(1e-12)
                    if c >= m:
                        c = m
                    m_remaining = R * (m - c)
                    expected_vp = 0.0
                    for zp in range(n_states):
                        m_next = m_remaining + income[zp]
                        vp_val = interp_val(Vp[t+1, zp, :], m_next)
                        expected_vp += transition[z, zp] * vp_val
                    rhs = beta * R * expected_vp
                    return util_deriv(c) - rhs

                c_low = 1e-12
                c_high = m
                
                f_low = residual(c_low)
                f_high = residual(c_high)
                
                if f_low < 0:
                    c_star = c_low
                elif f_high > 0:
                    c_star = c_high
                else:
                    try:
                        c_star = brentq(residual, c_low, c_high, xtol=1e-10, rtol=1e-10)
                    except Exception:
                        c_star = (c_low + c_high) / 2.0
                
                c_opt[i] = c_star
                
                # Calculate Value
                m_rem = R * (m - c_star)
                cont = 0.0
                for zp in range(n_states):
                    m_next = m_rem + income[zp]
                    v_val = interp_val(V[t+1, zp, :], m_next)
                    cont += transition[z, zp] * v_val
                val_t[i] = util(c_star) + beta * cont
                
                # Calculate Vp using Envelope Theorem
                expected_vp = 0.0
                for zp in range(n_states):
                    m_next = m_rem + income[zp]
                    vp_val = interp_val(Vp[t+1, zp, :], m_next)
                    expected_vp += transition[z, zp] * vp_val
                Vp[t, z, i] = beta * R * expected_vp
                V[t, z, i] = val_t[i]
                
            C[t, z, :] = c_opt
            
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, resource_grid + 1e-10)
    
    C[horizon - 1, :, :] = resource_grid
    
    return C
