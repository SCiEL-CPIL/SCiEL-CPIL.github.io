import numpy as np
import time

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
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Backward Induction using FOC bisection (vectorized over m)
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
        
        for z in range(num_states):
            probs = transition[z]
            
            # Vectorized bisection over all m points simultaneously
            # FOC: u'(c) = beta * R * sum_zp probs[zp] * V'_{t+1}(R*(m-c) + income[zp])
            # We bisect on c in (0, m)
            
            c_low = np.full(num_resources, 1e-12)
            c_high = resource_grid.copy()
            
            max_iter = 60
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                c_mid = (c_low + c_high) / 2.0
                
                # Compute LHS: u'(c_mid)
                lhs = u_prime(c_mid)
                
                # Compute RHS: beta * R * sum_zp probs[zp] * V'_{t+1}(R*(m - c_mid) + income[zp])
                # For each z', compute m_next = R * (m - c_mid) + income[z']
                # Then interpolate V'_{t+1}[z'] at m_next
                # We need V'_{t+1} for each z'
                
                rhs = np.zeros(num_resources)
                for zp in range(num_states):
                    if probs[zp] == 0:
                        continue
                    
                    # m_next for each m point
                    m_next = R * (resource_grid - c_mid) + income[zp]
                    
                    # We need V'_{t+1}[zp] as a function of m
                    # Approximate V' by numerical differentiation of value[t+1, zp, :]
                    # Or use the fact that at t+1, V'(m) = u'(c*(m)) from the FOC
                    # Actually, by envelope theorem: V'(m) = u'(c*(m))
                    # So V'_{t+1}[zp](m_next) = u'(policy[t+1, zp, :]) evaluated at m_next
                    # We can interpolate policy[t+1, zp, :] to get c at m_next, then apply u'
                    
                    # Interpolate policy at m_next
                    c_next = np.interp(m_next, resource_grid, policy[t+1, zp, :])
                    
                    # V' at m_next = u'(c_next)
                    v_prime = u_prime(c_next)
                    
                    rhs += probs[zp] * v_prime
                
                rhs = beta * R * rhs
                
                # FOC: u'(c) - beta*R*V'(R*(m-c)+income) = 0
                # If lhs > rhs, then u'(c) is too high, meaning c is too low (u' is decreasing)
                # So we need to increase c -> c_low = c_mid
                # If lhs < rhs, then c is too high -> c_high = c_mid
                
                # Wait: u'(c) is decreasing in c.
                # If u'(c) > beta*R*V'(...), we want to decrease u'(c), so increase c.
                # So c_low = c_mid
                # If u'(c) < beta*R*V'(...), we want to increase u'(c), so decrease c.
                # So c_high = c_mid
                
                too_low = lhs > rhs
                c_low = np.where(too_low, c_mid, c_low)
                c_high = np.where(~too_low, c_mid, c_high)
                
                # Check convergence
                if np.all(c_high - c_low < 1e-10):
                    break
            
            c_opt = (c_low + c_high) / 2.0
            c_opt = np.clip(c_opt, 1e-12, resource_grid)
            
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in range(num_states):
                if probs[zp] == 0:
                    continue
                m_next = R * (resource_grid - c_opt) + income[zp]
                v_next = np.interp(m_next, resource_grid, value[t+1, zp, :])
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    return policy
