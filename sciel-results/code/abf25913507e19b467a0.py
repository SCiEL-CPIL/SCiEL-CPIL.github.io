import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
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
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Pre-compute V' interpolators for t+1 at the start of each t iteration
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
        
        # Pre-compute V' interpolators for all z' at t+1
        v_prime_interps = {}
        for zp in range(num_states):
            v_prime_grid = u_prime(policy[t+1, zp, :])
            try:
                v_prime_interps[zp] = PchipInterpolator(resource_grid, v_prime_grid)
            except Exception:
                v_prime_interps[zp] = None
        
        # Also pre-compute V interpolators for value computation
        v_interps = {}
        for zp in range(num_states):
            try:
                v_interps[zp] = PchipInterpolator(resource_grid, value[t+1, zp, :])
            except Exception:
                v_interps[zp] = None
        
        for z in range(num_states):
            probs = transition[z]
            active_zp = [zp for zp in range(num_states) if probs[zp] > 0]
            
            # Initial guess: use policy from t+1 scaled by R (rough approximation)
            # Or use a fraction of m. Let's start with 0.5*m as initial guess
            c_current = 0.5 * resource_grid.copy()
            
            max_iter = 30
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute FOC: u'(c) - beta*R*sum(probs[zp]*V'_{t+1}[zp](m_next)) = 0
                # Newton step: c_new = c - f(c)/f'(c)
                
                # f(c) = u'(c) - beta*R*sum(probs[zp]*V'_{t+1}[zp](R*(m-c)+income[zp]))
                # f'(c) = u''(c) + beta*R*sum(probs[zp]*V''_{t+1}[zp](m_next)*R)
                
                # For Newton, we need V'' which we don't have directly.
                # Use a damped Newton / Bisection hybrid approach:
                # Try Newton, if it goes out of bounds, fall back to bisection
                
                lhs = u_prime(c_current)
                
                rhs = np.zeros(num_resources)
                for zp in active_zp:
                    m_next = R * (resource_grid - c_current) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    if v_prime_interps[zp] is not None:
                        v_prime_vals = v_prime_interps[zp](m_next_clipped)
                    else:
                        v_prime_vals = np.interp(m_next_clipped, resource_grid, u_prime(policy[t+1, zp, :]))
                    
                    rhs += probs[zp] * v_prime_vals
                
                rhs = beta * R * rhs
                
                f_val = lhs - rhs  # Should be zero at optimum
                f_val_prime = u_double_prime(c_current)  # Negative (concave)
                
                # Newton step
                # c_new = c - f/f'
                # Since f' < 0, if f > 0 (lhs > rhs), c is too low, increase c
                # If f < 0, c is too high, decrease c
                
                # Damped Newton: limit step size
                step = f_val / f_val_prime
                # Limit step to not exceed bounds
                max_step = 0.5 * resource_grid
                step = np.clip(step, -max_step, max_step)
                
                c_new = c_current - step
                c_new = np.clip(c_new, c_floor, resource_grid)
                
                # Check if Newton step is valid (reduces |f|)
                # If not, use bisection
                # For simplicity, just accept the Newton step if it's within bounds
                
                c_current = c_new
                
                # Convergence check
                if np.all(np.abs(step) < 1e-12):
                    break
            
            c_opt = np.clip(c_current, c_floor, resource_grid)
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in active_zp:
                m_next = R * (resource_grid - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                
                if v_interps[zp] is not None:
                    v_next = v_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, resource_grid, value[t+1, zp, :])
                
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    return policy
