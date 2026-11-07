"""Improved program: piecewise log-linear value function with analytical CRRA Euler solve."""
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Precompute utility
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Precompute log of resource grid for log-linear interpolation
    log_grid = np.log(resource_grid)
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    
    # Terminal condition: at t=horizon-1 (last period), consume all wealth
    # V_{H-1}(m, z) = u(m), c_{H-1}(m, z) = m
    V_next = np.zeros((n_states, n_grid))
    for z in range(n_states):
        V_next[z, :] = util(resource_grid)
    
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon-1, :, :] = resource_grid
    
    # Helper: interpolate V_next[zp] at given m_prime values using log-linear interp
    # Returns (value, derivative) where derivative is dV/dm
    def interp_loglinear(V_zp, m_prime_vals):
        """V_zp: array of shape (n_grid,), m_prime_vals: array of shape (n,)"""
        m_clipped = np.clip(m_prime_vals, m_min, m_max)
        log_m = np.log(m_clipped)
        
        # np.interp for values
        v_interp = np.interp(log_m, log_grid, V_zp)
        
        # Derivative: compute slope in log-space, then dV/dm = (dV/dlog_m) / m
        idx = np.searchsorted(log_grid, log_m)
        idx = np.clip(idx, 1, n_grid - 1)
        idx0 = idx - 1
        idx1 = idx
        
        dlog = log_grid[idx1] - log_grid[idx0]
        dV_dlog = (V_zp[idx1] - V_zp[idx0]) / dlog
        v_deriv = dV_dlog / m_clipped
        
        return v_interp, v_deriv
    
    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon-2, -1, -1):
        V_curr = np.zeros((n_states, n_grid))
        
        for z in range(n_states):
            m_vals = resource_grid
            
            # Solve Euler equation: mu(c) = beta * R * E_z'[V'(m')]
            # where m' = R*(m-c) + income[z']
            # For CRRA: mu(c) = c^{-gamma} (or 1/c for log)
            # So c = (beta * R * E[V'(m')])^{-1/gamma} for gamma != 1
            #    c = 1 / (beta * R * E[V'(m')]) for gamma = 1
            
            # Iterate to convergence (E depends on c through m')
            c_vals = np.clip(m_vals * 0.5, 1e-12, m_vals)
            
            for _ in range(20):
                # Compute E[V'(m')] for each m
                cont_deriv = np.zeros(n_grid)
                
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    
                    m_prime = R * (m_vals - c_vals) + income[zp]
                    _, v_deriv = interp_loglinear(V_next[zp], m_prime)
                    cont_deriv += p * v_deriv
                
                # Solve for c from mu(c) = beta * R * cont_deriv
                rhs = beta * R * cont_deriv
                rhs = np.maximum(rhs, 1e-30)
                
                if abs(gamma - 1.0) < 1e-10:
                    # mu(c) = 1/c => c = 1/rhs
                    c_new = 1.0 / rhs
                else:
                    # mu(c) = c^{-gamma} => c = rhs^{-1/gamma}
                    c_new = np.power(rhs, -1.0 / gamma)
                
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_vals)) < 1e-12 * np.max(m_vals):
                    c_vals = c_new
                    break
                c_vals = c_new
            
            consumption[t, z, :] = c_vals
            
            # Compute V_curr[z] = u(c) + beta * E[V(m')]
            cont_val = np.zeros(n_grid)
            for zp in range(n_states):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                m_prime = R * (m_vals - c_vals) + income[zp]
                v_interp, _ = interp_loglinear(V_next[zp], m_prime)
                cont_val += p * v_interp
            
            V_curr[z, :] = util(c_vals) + beta * cont_val
        
        V_next = V_curr
    
    # Ensure final period constraint
    consumption[horizon-1, :, :] = resource_grid
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid)
    
    return consumption
