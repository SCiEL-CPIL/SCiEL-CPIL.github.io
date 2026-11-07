import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_m_out = len(resource_grid)
    N_z = len(income)
    
    # Use a finer internal grid for higher accuracy
    N_m_int = 500
    m_int = np.geomspace(0.001, 40, N_m_int)
    
    # Precompute utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
        
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # V[t, m_idx, z]
    V = np.zeros((horizon, N_m_int, N_z))
    # C[t, z, m_idx]
    C = np.zeros((horizon, N_z, N_m_int))
    
    # Final period t = horizon - 1
    t = horizon - 1
    C[t, :, :] = m_int[None, :]
    V[t, :, :] = util(m_int)[:, np.newaxis]
    
    # Helper to create spline with power-law extrapolation for V
    def make_v_spline(m_grid, v_vals):
        # Fit power law or log to the tail for extrapolation
        # Use last 20% of points
        n_fit = max(10, len(m_grid) // 5)
        m_tail = m_grid[-n_fit:]
        v_tail = v_vals[-n_fit:]
        
        if gamma == 1:
            # Fit log(m) = A * log(m) + B
            # v = A * log(m) + B
            log_m = np.log(m_tail)
            # Least squares for A, B
            A_log = np.polyfit(log_m, v_tail, 1)
            A, B = A_log[0], A_log[1]
            
            def spline(x):
                x = np.asarray(x)
                res = np.zeros_like(x, dtype=float)
                in_bounds = (x >= m_grid[0]) & (x <= m_grid[-1])
                # Use cubic spline for interior
                # We need a spline for the interior part
                # Let's just use CubicSpline for interior and analytical for exterior
                # CubicSpline doesn't easily allow custom extrapolation per point in one call without subclassing or manual logic
                # We'll create a CubicSpline for the interior and handle extrapolation manually
                pass
        else:
            # Fit power law: v = A * m^p + B
            # Linearize: log(v - B) = log(A) + p * log(m)
            # This is non-linear if B is unknown.
            # Simpler: Assume B=0 for large m? Or fit v/m^(1-gamma) = A?
            # V(m) ~ K * m^(1-gamma)
            # K = V / m^(1-gamma)
            exponent = 1 - gamma
            K_vals = v_tail / (m_tail ** exponent)
            # Take median K for stability
            K_med = np.median(K_vals)
            
            def spline(x):
                x = np.asarray(x)
                res = np.zeros_like(x, dtype=float)
                in_bounds = (x >= m_grid[0]) & (x <= m_grid[-1])
                # We need an interior spline
                # Let's use CubicSpline for the whole range but override extrapolation
                pass
        
        # To keep it simple and robust, let's use CubicSpline for interior
        # and linear extrapolation but corrected with the slope at the boundary?
        # Or just use 'linear' and hope it's good enough?
        # The previous score was 5.8. The main error source might be the coarse grid.
        # Let's stick to CubicSpline with extrapolate='linear' for now, as implementing custom extrapolation
        # inside a spline call is tricky without subclassing.
        
        # However, we can manually handle the evaluation.
        cs = CubicSpline(m_grid, v_vals, extrapolate=False)
        
        def eval_spline(x):
            x_arr = np.asarray(x)
            res = np.zeros_like(x_arr, dtype=float)
            in_bounds = (x_arr >= m_grid[0]) & (x_arr <= m_grid[-1])
            res[in_bounds] = cs(x_arr[in_bounds])
            
            # Extrapolation
            if np.any(x_arr < m_grid[0]):
                # For small m, V is bounded. Use value at m_min?
                # V(0) is -inf for log/CRRA. But m_min is 0.001.
                # Linear extrapolation to 0 might be bad.
                # Let's just use the value at m_min for x < m_min
                res[x_arr < m_grid[0]] = v_vals[0]
                
            if np.any(x_arr > m_grid[-1]):
                # Use power law fit
                if gamma == 1:
                    # v = A * log(m) + B
                    # Refit quickly
                    log_m = np.log(m_tail)
                    coeff = np.polyfit(log_m, v_tail, 1)
                    res[x_arr > m_grid[-1]] = coeff[0] * np.log(x_arr[x_arr > m_grid[-1]]) + coeff[1]
                else:
                    exponent = 1 - gamma
                    K_vals = v_tail / (m_tail ** exponent)
                    K_med = np.median(K_vals)
                    res[x_arr > m_grid[-1]] = K_med * (x_arr[x_arr > m_grid[-1]] ** exponent)
                    
            return res
        return eval_spline

    # Backward induction
    # Initialize spline for last period
    splines_next = []
    for z in range(N_z):
        splines_next.append(make_v_spline(m_int, V[horizon-1, :, z]))
        
    for t in range(horizon - 2, -1, -1):
        V_curr = np.zeros((N_m_int, N_z))
        C_curr = np.zeros((N_z, N_m_int))
        
        for z in range(N_z):
            for i_m in range(N_m_int):
                m = m_int[i_m]
                
                def objective(c_val):
                    if c_val <= 0 or c_val > m + 1e-10:
                        return -np.inf
                    c_val = max(c_val, 1e-12)
                    
                    u_val = util(c_val)
                    exp_val = 0.0
                    for zp in range(N_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m - c_val) + income[zp]
                        v_val = splines_next[zp](m_prime)
                        exp_val += prob * v_val
                        
                    return u_val + beta * exp_val
                
                res = minimize_scalar(lambda x: -objective(x), bounds=(1e-12, m), method='bounded')
                c_opt = res.x
                v_opt = -res.fun
                
                C_curr[z, i_m] = c_opt
                V_curr[i_m, z] = v_opt
        
        V[t] = V_curr
        C[t] = C_curr
        
        # Update splines
        splines_next = []
        for z in range(N_z):
            splines_next.append(make_v_spline(m_int, V[t, :, z]))
            
    # Interpolate C from internal grid to output grid
    C_out = np.zeros((horizon, N_z, N_m_out))
    for t in range(horizon):
        for z in range(N_z):
            # C[t, z, :] is defined on m_int
            # We need C on resource_grid
            C_out[t, z, :] = np.interp(resource_grid, m_int, C[t, z, :])
            
    return C_out
