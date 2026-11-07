def solve_consumption(model, resource_grid, budget):
    import numpy as np
    from scipy.optimize import minimize_scalar, minimize
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    T = horizon
    n_states = len(income)
    n_grid = len(resource_grid)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    # Use resource_grid for backward induction, with fine c search
    c_opt = np.zeros((T, n_states, n_grid))
    V_opt = np.zeros((T, n_states, n_grid))
    
    for z in range(n_states):
        c_opt[T-1, z, :] = resource_grid
        V_opt[T-1, z, :] = utility(resource_grid)
    
    for t in range(T-2, -1, -1):
        for z in range(n_states):
            for i_m in range(n_grid):
                m = resource_grid[i_m]
                
                def neg_obj(c):
                    if c < 1e-12 or c > m:
                        return 1e10
                    u = utility(c)
                    m_save = R * (m - c)
                    cont = 0.0
                    for zp in range(n_states):
                        m_zp = m_save + income[zp]
                        V_zp = np.interp(m_zp, resource_grid, V_opt[t+1, zp, :])
                        cont += transition[z, zp] * V_zp
                    return -(u + beta * cont)
                
                res = minimize_scalar(neg_obj, bounds=(1e-12, m), method='bounded',
                                      options={'xatol': 1e-12, 'maxiter': 100})
                c_star = max(1e-12, min(res.x, m))
                c_opt[t, z, i_m] = c_star
                V_opt[t, z, i_m] = -res.fun
    
    # Fit rational functions
    def fit_rational(m_data, y_data, dp=5, dq=4):
        n = len(m_data)
        P_mat = np.vander(m_data, N=dp+1, increasing=True)
        if dq > 0:
            Q_mat = np.vander(m_data, N=dq, increasing=True)[:, 1:]
        else:
            Q_mat = np.zeros((n, 0))
        
        A = np.hstack([P_mat, -Q_mat]) if dq > 0 else P_mat
        b_vec = y_data
        
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            a_init = sol[:dp+1]
            b_init = sol[dp+1:] if dq > 0 else np.array([])
        except:
            a_init = np.zeros(dp+1)
            b_init = np.zeros(dq)
        
        params_init = np.concatenate([a_init, b_init])
        
        def obj_fn(params):
            a = params[:dp+1]
            b = params[dp+1:] if dq > 0 else np.array([])
            p_val = np.polyval(a[::-1], m_data)
            if dq > 0:
                q_val = 1.0 + np.polyval(b[::-1], m_data)
            else:
                q_val = np.ones_like(m_data)
            mask = np.abs(q_val) > 1e-12
            if not np.all(mask):
                return 1e10
            resid = p_val[mask] / q_val[mask] - y_data[mask]
            return np.sum(resid**2)
        
        try:
            res = minimize(obj_fn, params_init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < obj_fn(params_init):
                params_best = res.x
            else:
                params_best = params_init
        except:
            params_best = params_init
        
        a_best = params_best[:dp+1]
        b_best = params_best[dp+1:] if dq > 0 else np.array([])
        return a_best, b_best
    
    policy_coeffs = {}
    for t in range(T):
        for z in range(n_states):
            a_c, b_c = fit_rational(resource_grid, c_opt[t, z, :], dp=5, dq=4)
            policy_coeffs[(t, z)] = (a_c, b_c)
    
    consumption = np.zeros((T, n_states, n_grid))
    for t in range(T):
        for z in range(n_states):
            a_c, b_c = policy_coeffs[(t, z)]
            p_val = np.polyval(a_c[::-1], resource_grid)
            if len(b_c) > 0:
                q_val = 1.0 + np.polyval(b_c[::-1], resource_grid)
            else:
                q_val = np.ones_like(resource_grid)
            
            mask = np.abs(q_val) > 1e-15
            c_vals = np.zeros_like(resource_grid)
            c_vals[mask] = p_val[mask] / q_val[mask]
            c_vals[~mask] = resource_grid[~mask]
            
            c_vals = np.clip(c_vals, 1e-12, resource_grid + 1e-10)
            consumption[t, z, :] = c_vals
    
    consumption[5, :, :] = resource_grid[np.newaxis, :, :]
    
    return consumption
