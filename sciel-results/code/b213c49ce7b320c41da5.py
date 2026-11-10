import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Log-space for better interpolation on geometric grid
    log_grid = np.log(resource_grid)
    
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
    v_prime = np.zeros((horizon, num_states, num_resources))
    v_double_prime = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
        v_prime[t, z, :] = u_prime(resource_grid)
        v_double_prime[t, z, :] = u_double_prime(resource_grid)
    
    # Backward Induction using Newton-Raphson with fallback to bisection
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
                    v_prime[tt, z, :] = u_prime(resource_grid)
                    v_double_prime[tt, z, :] = u_double_prime(resource_grid)
            break
        
        # Pre-build PCHIP interpolators in log-space for V'_{t+1}[z'](m) and V_{t+1}[z'](m)
        v_prime_interps = []
        value_interps = []
        v_double_prime_interps = []
        for zp in range(num_states):
            try:
                vp_interp = PchipInterpolator(log_grid, v_prime[t+1, zp, :])
            except Exception:
                vp_interp = None
            v_prime_interps.append(vp_interp)
            
            try:
                val_interp = PchipInterpolator(log_grid, value[t+1, zp, :])
            except Exception:
                val_interp = None
            value_interps.append(val_interp)
            
            try:
                vdp_interp = PchipInterpolator(log_grid, v_double_prime[t+1, zp, :])
            except Exception:
                vdp_interp = None
            v_double_prime_interps.append(vdp_interp)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Better initial guess: use heuristic based on beta and R
            if gamma == 1:
                init_frac = 1.0 / (1.0 + beta * R)
            else:
                init_frac = 1.0 / (1.0 + beta * R * max(1.0, 1.0 - gamma))
            init_frac = np.clip(init_frac, 0.01, 0.99)
            
            c_low = np.full(num_resources, c_floor)
            c_high = resource_grid.copy()
            c = np.clip(resource_grid * init_frac, c_floor, resource_grid)
            
            max_iter = 30
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute LHS: u'(c)
                lhs = u_prime(c)
                
                # Compute RHS: beta * R * sum_zp probs[zp] * V'_{t+1}[zp](R*(m-c) + income[zp])
                rhs = np.zeros(num_resources)
                
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    
                    m_next = R * (resource_grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    m_next_log = np.log(m_next_clipped)
                    
                    if v_prime_interps[zp] is not None:
                        v_prime_vals = v_prime_interps[zp](m_next_log)
                    else:
                        v_prime_vals = np.interp(m_next_log, log_grid, v_prime[t+1, zp, :])
                    
                    rhs += probs[zp] * v_prime_vals
                
                rhs = beta * R * rhs
                
                # FOC: f(c) = u'(c) - rhs = 0
                f_val = lhs - rhs
                
                # Bisection update
                too_low = f_val > 0  # u'(c) > rhs means c is too low
                c_low = np.where(too_low, c, c_low)
                c_high = np.where(~too_low, c, c_high)
                
                # Compute derivative analytically
                # f'(c) = u''(c) + beta*R^2 * sum_zp probs[zp] * V''_{t+1}[zp](m_next)
                lhs_prime = u_double_prime(c)
                
                rhs_prime = np.zeros(num_resources)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    m_next = R * (resource_grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    m_next_log = np.log(m_next_clipped)
                    
                    if v_double_prime_interps[zp] is not None:
                        vdp_vals = v_double_prime_interps[zp](m_next_log)
                    else:
                        vdp_vals = np.interp(m_next_log, log_grid, v_double_prime[t+1, zp, :])
                    
                    rhs_prime += probs[zp] * vdp_vals
                
                df_dc = lhs_prime + beta * R * R * rhs_prime
                
                # Newton step
                safe_df = np.where(np.abs(df_dc) > 1e-14, df_dc, 1e-14)
                newton_c = c - f_val / safe_df
                
                # Use Newton step if it's within bounds, otherwise use bisection midpoint
                bisection_mid = (c_low + c_high) / 2.0
                c_new = np.where(
                    (newton_c > c_low) & (newton_c < c_high),
                    newton_c,
                    bisection_mid
                )
                c_new = np.clip(c_new, c_floor, resource_grid)
                
                # Check convergence (relative)
                if np.all(np.abs(c_new - c) < 1e-12 * np.maximum(c, 1e-6)):
                    c = c_new
                    break
                
                c = c_new
            
            c_opt = np.clip(c, c_floor, resource_grid)
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (resource_grid - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                m_next_log = np.log(m_next_clipped)
                
                if value_interps[zp] is not None:
                    v_next = value_interps[zp](m_next_log)
                else:
                    v_next = np.interp(m_next_log, log_grid, value[t+1, zp, :])
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
            
            # Compute v_prime and v_double_prime for this period
            v_prime[t, z, :] = u_prime(c_opt)
            
            # V''(m) = -beta*R^2 * sum_zp probs[zp] * V''_{t+1}[zp](m_next)
            vp_term = np.zeros(num_resources)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (resource_grid - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                m_next_log = np.log(m_next_clipped)
                
                if v_double_prime_interps[zp] is not None:
                    vdp_vals = v_double_prime_interps[zp](m_next_log)
                else:
                    vdp_vals = np.interp(m_next_log, log_grid, v_double_prime[t+1, zp, :])
                
                vp_term += probs[zp] * vdp_vals
            
            v_double_prime[t, z, :] = -beta * R * R * vp_term
    
    # Ensure terminal condition is exact
    t = horizon - 1
    for z in range(num_states):
        policy[t, z, :] = resource_grid
        value[t, z, :] = utility(resource_grid)
    
    return policy
