"""Rewritten solution for consumption-saving policy accuracy task."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
    
    # Backward induction
    # V[t, z, i] = value function at time t, state z, resource index i
    V = np.zeros((T, Z, N))
    
    # Terminal condition already handled in c_policy
    
    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Fallback: use simple heuristic for remaining periods
            for tt in range(t, T - 1):
                for z in range(Z):
                    c_policy[tt, z, :] = resource_grid * 0.5
            break
        
        for z in range(Z):
            # For each current state z, compute optimal consumption
            # Euler equation: u'(c) = beta * sum_{z'} P(z'|z) * u'(c') * R
            # where c' is next period consumption given m' = R*(m-c) + income[z']
            
            # We need to find c for each m in resource_grid
            # Use Newton's method on the FOC
            
            # Precompute next period values
            # For each possible next state z', the continuation value depends on m'
            # m' = R*(m - c) + income[z']
            
            # We'll use a smooth approximation approach
            # First, compute the continuation value function V_next(z', m') 
            # as a function of m' using the grid values and interpolation
            
            # For each z', we have V[t+1, z', :] as a function of resource_grid
            # We need to evaluate this at m' = R*(m - c) + income[z']
            
            # Strategy: use the Euler equation and solve for c
            # u'(c) = beta * sum_{z'} P(z'|z) * u'(c_{z'}) * R
            # where c_{z'} is next period consumption when next state is z'
            # and next period resources are m_{z'} = R*(m - c) + income[z']
            
            # This is implicit because c_{z'} depends on m_{z'} which depends on c
            # We can iterate: start with c = m (consume all), then update
            
            # Better approach: use the fact that in the interior, 
            # c_{z'} = argmax_{c'} [ u(c') + beta * V_next(z', m_{z'}) ]
            # But V_next already incorporates the optimization
            
            # Let's use direct maximization with fine grid for accuracy
            
            # Use a denser grid for consumption search
            c_fine_frac = np.linspace(0.001, 1.0, 201)  # 201 points
            
            for i in range(N):
                m = resource_grid[i]
                
                # Try different consumption levels
                c_candidates = m * c_fine_frac
                c_candidates = np.maximum(c_candidates, 1e-12)
                
                # For each c candidate, compute the objective
                # u(c) + beta * sum_{z'} P(z'|z) * V_next(z', m_{z'})
                # where m_{z'} = R*(m - c) + income[z']
                
                best_obj = -np.inf
                best_c = c_candidates[0]
                
                for z_prime in range(Z):
                    prob = transition[z, z_prime]
                    if prob < 1e-12:
                        continue
                    
                    m_prime = R * (m - c_candidates) + income[z_prime]
                    m_prime = np.maximum(m_prime, 0.0)
                    
                    # Interpolate V_next at m_prime
                    # resource_grid is the grid for V_next
                    V_interp = np.interp(m_prime, resource_grid, V[t+1, z_prime, :])
                
                # Compute objective for each c candidate
                u_c = util(c_candidates)
                
                # Sum over z' of prob * V_interp
                cont_value = np.zeros_like(c_candidates)
                for z_prime in range(Z):
                    prob = transition[z, z_prime]
                    if prob < 1e-12:
                        continue
                    m_prime = R * (m - c_candidates) + income[z_prime]
                    m_prime = np.maximum(m_prime, 0.0)
                    V_interp = np.interp(m_prime, resource_grid, V[t+1, z_prime, :])
                    cont_value += prob * V_interp
                
                obj = u_c + beta * cont_value
                idx = np.argmax(obj)
                best_c = c_candidates[idx]
                
                c_policy[t, z, i] = best_c
                V[t, z, i] = obj[idx]
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Ensure last period consumes all
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
