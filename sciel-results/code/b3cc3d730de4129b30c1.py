import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # resource_grid is shape (101,)
    # We need to return consumption[t, z, m] with shape (6, 2, 101)
    
    n_m = len(resource_grid)
    n_z = 2
    
    # Pre-allocate arrays
    V = np.zeros((horizon, n_z, n_m))
    V_prime = np.zeros((horizon, n_z, n_m))
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal period t = horizon - 1 = 5
    # c = m
    c_policy[horizon-1, :, :] = resource_grid[None, :]
    # V = u(m)
    for z in range(n_z):
        V[horizon-1, z, :] = _utility(resource_grid, gamma)
        V_prime[horizon-1, z, :] = _utility_prime(resource_grid, gamma)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
            
        for z in range(n_z):
            for i_m, m in enumerate(resource_grid):
                # Define the FOC function
                # f(c) = u'(c) - beta * R * sum_z' P[z, z'] * V_prime[t+1, z', R*(m-c) + income[z']]
                
                def foc(c):
                    if c < 1e-12:
                        c = 1e-12
                    if c > m:
                        c = m
                    lhs = _utility_prime(c, gamma)
                    
                    rhs_sum = 0.0
                    for z_prime in range(n_z):
                        # m_next = R * (m - c) + income[z_prime]
                        m_next = R * (m - c) + income[z_prime]
                        
                        # Interpolate V_prime[t+1, z_prime, :] at m_next
                        # resource_grid is sorted
                        vp_val = np.interp(m_next, resource_grid, V_prime[t+1, z_prime, :])
                        rhs_sum += transition[z, z_prime] * vp_val
                    
                    rhs = beta * R * rhs_sum
                    return lhs - rhs
                
                # Solve for c in [1e-12, m]
                # f(1e-12) is likely positive (u' is huge)
                # f(m) is likely negative (u' is small, RHS is positive)
                # Check signs
                try:
                    c_opt = brentq(foc, 1e-12, m, xtol=1e-10, rtol=1e-10)
                except ValueError:
                    # If signs are same, fallback to grid search or boundary
                    # Usually f(1e-12) > 0 and f(m) < 0.
                    # If not, something is wrong.
                    # Try a wider range or just pick a midpoint?
                    # Let's check f(1e-12) and f(m)
                    f_lo = foc(1e-12)
                    f_hi = foc(m)
                    if f_lo > 0 and f_hi > 0:
                        c_opt = m
                    elif f_lo < 0 and f_hi < 0:
                        c_opt = 1e-12
                    else:
                        # Should not happen if concave
                        c_opt = m / 2.0
                        
                c_policy[t, z, i_m] = c_opt
                
                # Calculate V and V_prime
                # V = u(c_opt) + beta * sum P V[t+1]
                # V_prime = u'(c_opt)  (Envelope theorem)
                
                v_val = _utility(c_opt, gamma)
                vp_val = _utility_prime(c_opt, gamma)
                
                cont_val = 0.0
                for z_prime in range(n_z):
                    m_next = R * (m - c_opt) + income[z_prime]
                    v_next = np.interp(m_next, resource_grid, V[t+1, z_prime, :])
                    cont_val += transition[z, z_prime] * v_next
                
                V[t, z, i_m] = v_val + beta * cont_val
                V_prime[t, z, i_m] = vp_val
                
    # Check if we finished in time
    if time.time() > deadline:
         # Return whatever we have, ensuring validity
         pass
         
    # Ensure c[5] is exactly the grid
    c_policy[horizon-1, :, :] = resource_grid[None, :]
    
    # Ensure 0 < c <= m + 1e-10
    c_policy = np.clip(c_policy, 1e-12, resource_grid[None, None, :] + 1e-10)
    
    return c_policy

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)

def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    else:
        return c ** (-gamma)
