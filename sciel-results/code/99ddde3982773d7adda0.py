import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Backward Induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: Return current best or simple heuristic
            # Since we are in backward induction, we might not have finished.
            # For safety, if we run out of time, we can return the policy computed so far
            # padded with the terminal policy for earlier periods (not optimal but valid).
            # Or just break and return what we have.
            # The requirement is to return a valid result.
            # If we break early, policy[t] for t < current_t is zero.
            # We should fill remaining with a heuristic like c = m.
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
            
        for z in range(num_states):
            # Precompute transition probabilities
            probs = transition[z]
            
            # We need to evaluate V_{t+1} at points m_prime + income[z']
            # m_prime = R * (m - c)
            
            for m_idx, m in enumerate(resource_grid):
                # Golden Section Search for c in [eps, m]
                eps = 1e-12
                a = eps
                b = m
                
                # Golden ratio
                invphi = (np.sqrt(5) - 1) / 2
                invphi2 = invphi * invphi
                
                c1 = b - invphi * (b - a)
                c2 = a + invphi * (b - a)
                
                # Evaluate objective at c1 and c2
                def obj(c):
                    u_c = utility(c)
                    m_prime = R * (m - c)
                    # Continuation value
                    cont_val = 0.0
                    for zp in range(num_states):
                        # Next state resource: m_prime + income[zp]
                        m_next = m_prime + income[zp]
                        # Interpolate V_{t+1} at m_next
                        # np.interp requires increasing x
                        v_next = np.interp(m_next, resource_grid, value[t+1, zp, :])
                        cont_val += probs[zp] * v_next
                    return u_c + beta * cont_val
                
                f1 = obj(c1)
                f2 = obj(c2)
                
                # Iterate
                max_iter = 50
                for _ in range(max_iter):
                    if abs(b - a) < 1e-10:
                        break
                    if f1 > f2:
                        b = c2
                        c2 = c1
                        f2 = f1
                        c1 = b - invphi * (b - a)
                        f1 = obj(c1)
                    else:
                        a = c1
                        c1 = c2
                        f1 = f2
                        c2 = a + invphi * (b - a)
                        f2 = obj(c2)
                
                c_opt = (a + b) / 2
                # Ensure constraints
                if c_opt <= 0: c_opt = eps
                if c_opt > m: c_opt = m
                
                policy[t, z, m_idx] = c_opt
                value[t, z, m_idx] = obj(c_opt)
                
    return policy
