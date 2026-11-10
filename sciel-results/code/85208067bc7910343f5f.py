def solve_consumption(model, resource_grid, budget):
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    log_m = np.log(resource_grid)
    
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    V_splines = [None] * horizon
    c_policy = np.zeros((horizon, n_states, n_grid))
    
    # Terminal: t = horizon - 1
    t_term = horizon - 1
    V_splines[t_term] = [None] * n_states
    for z in range(n_states):
        V_vals = u(resource_grid)
        V_splines[t_term][z] = CubicSpline(log_m, V_vals, bc_type='natural')
    for z in range(n_states):
        c_policy[t_term, z, :] = resource_grid
    
    # Backward induction
    for t in range(t_term - 1, -1, -1):
        V_splines[t] = [None] * n_states
        for z in range(n_states):
            c_vals = np.zeros(n_grid)
            V_vals = np.zeros(n_grid)
            for i in range(n_grid):
                m = resource_grid[i]
                
                def euler_f(c):
                    c = max(c, 1e-12)
                    c = min(c, m - 1e-12)
                    if c >= m - 1e-12:
                        c = m - 1e-12
                    cont = 0.0
                    for zp in range(n_states):
                        m_next = R * (m - c) + income[zp]
                        m_next = max(m_next, 1e-12)
                        dV_dlogm = V_splines[t+1][zp].derivative(1)(np.log(m_next))
                        dV_dm = dV_dlogm / m_next
                        cont += transition[z, zp] * dV_dm
                    return u_prime(c) - beta * cont
                
                c_lo = 1e-12
                c_hi = m - 1e-12
                if c_hi <= c_lo:
                    c_opt = m / 2.0
                else:
                    f_lo = euler_f(c_lo)
                    f_hi = euler_f(c_hi)
                    if f_lo <= 0:
                        c_opt = c_lo
                    elif f_hi >= 0:
                        c_opt = c_hi
                    else:
                        try:
                            c_opt = brentq(euler_f, c_lo, c_hi, xtol=1e-14, rtol=1e-14, maxiter=100)
                        except Exception:
                            a, b = c_lo, c_hi
                            for _ in range(80):
                                mid = (a + b) / 2.0
                                if euler_f(mid) > 0:
                                    a = mid
                                else:
                                    b = mid
                            c_opt = (a + b) / 2.0
                
                c_vals[i] = c_opt
                cont_val = 0.0
                for zp in range(n_states):
                    m_next = R * (m - c_opt) + income[zp]
                    m_next = max(m_next, 1e-12)
                    cont_val += transition[z, zp] * V_splines[t+1][zp](np.log(m_next))
                V_vals[i] = u(c_opt) + beta * cont_val
            
            c_policy[t, z, :] = c_vals
            V_splines[t][z] = CubicSpline(log_m, V_vals, bc_type='natural')
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_states):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    return c_policy
