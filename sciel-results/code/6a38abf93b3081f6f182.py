import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Utility second derivative
    def u_double_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1)
    
    # Policy and Value arrays
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Backward Induction using Newton-Raphson with PCHIP interpolation
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
        
        # Pre-build interpolators for V'_{t+1}[z'](m) for all z'
        v_prime_interps = []
        value_interps = []
        for zp in range(num_states):
            v_prime_grid = u_prime(policy[t + 1, zp, :])
            try:
                vp_interp = PchipInterpolator(resource_grid, v_prime_grid)
            except Exception:
                vp_interp = None
            v_prime_interps.append(vp_interp)
            
            try:
                val_interp = PchipInterpolator(resource_grid, value[t + 1, zp, :])
            except Exception:
                val_interp = None
            value_interps.append(val_interp)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Initialize with previous period's policy as starting guess
            c_current = policy[t + 1, z, :].copy()
            c_current = np.clip(c_current, c_floor, resource_grid)
            
            max_iter = 30
            converged = False
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute FOC residual: f(c) = u'(c) - beta*R*sum(probs[z']*V'_{t+1}[z'](R*(m-c)+income[z']))
                c_safe = np.maximum(c_current, c_floor)
                lhs = u_prime(c_safe)
                
                rhs = np.zeros(num_resources)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    m_next = R * (resource_grid - c_safe) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    if v_prime_interps[zp] is not None:
                        v_prime_vals = v_prime_interps[zp](m_next_clipped)
                    else:
                        v_prime_grid = u_prime(policy[t + 1, zp, :])
                        v_prime_vals = np.interp(m_next_clipped, resource_grid, v_prime_grid)
                    
                    rhs += probs[zp] * v_prime_vals
                
                rhs = beta * R * rhs
                residual = lhs - rhs
                
                # Check convergence
                if np.all(np.abs(residual) < 1e-10):
                    converged = True
                    break
                
                # Newton step: f'(c) = u''(c) + beta*R*sum(probs[z']*V''_{t+1}[z'](...)*R)
                # Approximate V'' numerically using finite differences on the interpolator
                f_prime = u_double_prime(c_safe)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    m_next = R * (resource_grid - c_safe) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    # Numerical second derivative of V'
                    h = 1e-5
                    m_next_p = np.clip(m_next_clipped + h, resource_grid[0], resource_grid[-1])
                    m_next_m = np.clip(m_next_clipped - h, resource_grid[0], resource_grid[-1])
                    
                    if v_prime_interps[zp] is not None:
                        vp_p = v_prime_interps[zp](m_next_p)
                        vp_m = v_prime_interps[zp](m_next_m)
                    else:
                        v_prime_grid = u_prime(policy[t + 1, zp, :])
                        vp_p = np.interp(m_next_p, resource_grid, v_prime_grid)
                        vp_m = np.interp(m_next_m, resource_grid, v_prime_grid)
                    
                    v_dd_prime = (vp_p - vp_m) / (2 * h)
                    f_prime += beta * R * probs[zp] * v_dd_prime * R
                
                # Newton update with damping
                step = residual / np.maximum(np.abs(f_prime), 1e-15)
                c_new = c_current - step
                
                # Damping: ensure c stays in [c_floor, m]
                c_new = np.clip(c_new, c_floor, resource_grid)
                
                # Check if step is too large, damp it
                too_large = np.abs(c_new - c_current) > 0.5 * resource_grid
                c_new = np.where(too_large, c_current - 0.5 * step, c_new)
                c_new = np.clip(c_new, c_floor, resource_grid)
                
                c_current = c_new
            
            # If Newton didn't converge, fall back to bisection for non-converged points
            if not converged:
                c_low = np.full(num_resources, c_floor)
                c_high = resource_grid.copy()
                # Use c_current as initial bracket if possible
                for i in range(num_resources):
                    if c_current[i] > c_low[i] and c_current[i] < c_high[i]:
                        # Evaluate residual at c_current
                        c_safe_i = max(c_current[i], c_floor)
                        lhs_i = u_prime(np.array([c_safe_i]))[0]
                        rhs_i = 0.0
                        for zp in range(num_states):
                            if probs[zp] == 0:
                                continue
                            m_next_i = R * (resource_grid[i] - c_safe_i) + income[zp]
                            m_next_clipped_i = np.clip(m_next_i, resource_grid[0], resource_grid[-1])
                            if v_prime_interps[zp] is not None:
                                vp_i = v_prime_interps[zp](np.array([m_next_clipped_i]))[0]
                            else:
                                v_prime_grid = u_prime(policy[t + 1, zp, :])
                                vp_i = np.interp(m_next_clipped_i, resource_grid, v_prime_grid)
                            rhs_i += probs[zp] * vp_i
                        rhs_i = beta * R * rhs_i
                        res_i = lhs_i - rhs_i
                        if res_i > 0:
                            c_low[i] = c_current[i]
                        else:
                            c_high[i] = c_current[i]
                
                for _ in range(50):
                    c_mid = (c_low + c_high) / 2.0
                    c_safe = np.maximum(c_mid, c_floor)
                    lhs = u_prime(c_safe)
                    rhs = np.zeros(num_resources)
                    for zp in range(num_states):
                        if probs[zp] == 0:
                            continue
                        m_next = R * (resource_grid - c_safe) + income[zp]
                        m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                        if v_prime_interps[zp] is not None:
                            v_prime_vals = v_prime_interps[zp](m_next_clipped)
                        else:
                            v_prime_grid = u_prime(policy[t + 1, zp, :])
                            v_prime_vals = np.interp(m_next_clipped, resource_grid, v_prime_grid)
                        rhs += probs[zp] * v_prime_vals
                    rhs = beta * R * rhs
                    too_low = lhs > rhs
                    c_low = np.where(too_low, c_mid, c_low)
                    c_high = np.where(~too_low, c_mid, c_high)
                    if np.all(c_high - c_low < 1e-12):
                        break
                c_current = (c_low + c_high) / 2.0
            
            c_opt = np.clip(c_current, c_floor, resource_grid)
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (resource_grid - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                
                if value_interps[zp] is not None:
                    v_next = value_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, resource_grid, value[t + 1, zp, :])
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    # Ensure terminal condition is exact
    t = horizon - 1
    for z in range(num_states):
        policy[t, z, :] = resource_grid
        value[t, z, :] = utility(resource_grid)
    
    return policy
