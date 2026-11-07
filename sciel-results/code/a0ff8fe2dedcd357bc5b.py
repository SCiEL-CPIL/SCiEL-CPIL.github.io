import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 150s
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    n_out = len(resource_grid)
    
    # Internal fine grid for value function and policy calculation
    # Use a grid finer than the 251 scoring grid to ensure interpolation accuracy
    n_fine = 500
    m_fine = np.geomspace(0.001, 40, n_fine)
    
    # Utility and Derivative functions
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def utility_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Initialize Value Function and Derivative for terminal period (t = horizon)
    # V_next has shape (n_fine, 2)
    V_next = np.zeros((n_fine, 2))
    V_prime_next = np.zeros((n_fine, 2))
    
    # Policy arrays for fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Backward Induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5: # 5s buffer
            break
            
        V_curr = np.zeros((n_fine, 2))
        V_prime_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            probs = transition[z]
            
            for i in range(n_fine):
                m = m_fine[i]
                
                # Define the residual function for the FOC
                # g(c) = U'(c) - beta * sum_{z'} P_{z,z'} * R * V'_{t+1}(R(m-c) + y_{z'}, z')
                
                def residual(c):
                    if c < 1e-12:
                        c = 1e-12
                    if c > m:
                        c = m
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Interpolate V_prime_next
                    # V_prime_next[:, 0] and V_prime_next[:, 1] are the derivatives on m_fine
                    vp0 = np.interp(m_prime_0, m_fine, V_prime_next[:, 0])
                    vp1 = np.interp(m_prime_1, m_fine, V_prime_next[:, 1])
                    
                    ev_prime = probs[0] * vp0 + probs[1] * vp1
                    return utility_prime(c) - beta * R * ev_prime

                # Find root in [1e-12, m]
                # Check signs at boundaries
                c_min = 1e-12
                c_max = m
                
                # If m is very small, c_min might be >= c_max
                if c_min >= c_max:
                    c_opt = c_max
                else:
                    f_min = residual(c_min)
                    f_max = residual(c_max)
                    
                    # If no sign change, we might be at a boundary
                    # For concave utility, if f_min < 0, optimal c is c_min (corner)
                    # If f_max > 0, optimal c is c_max (corner)
                    # Otherwise, root exists.
                    
                    if f_min * f_max < 0:
                        try:
                            c_opt = brentq(residual, c_min, c_max, xtol=1e-12, rtol=1e-12, maxiter=50)
                        except ValueError:
                            # Fallback to grid search if brentq fails
                            # Coarse grid
                            c_grid = np.linspace(c_min, c_max, 100)
                            vals = [residual(c) for c in c_grid]
                            # Find sign change
                            for j in range(len(c_grid)-1):
                                if vals[j] * vals[j+1] < 0:
                                    try:
                                        c_opt = brentq(residual, c_grid[j], c_grid[j+1], xtol=1e-12, rtol=1e-12)
                                    except:
                                        c_opt = c_grid[j] if abs(vals[j]) < abs(vals[j+1]) else c_grid[j+1]
                                    break
                            else:
                                # No sign change found in grid, pick boundary with smaller absolute residual
                                c_opt = c_min if abs(f_min) < abs(f_max) else c_max
                    else:
                        # No sign change
                        if f_min < 0:
                            c_opt = c_min
                        elif f_max > 0:
                            c_opt = c_max
                        else:
                            # Should not happen for concave problem, but handle gracefully
                            c_opt = c_min if abs(f_min) < abs(f_max) else c_max

                # Ensure c_opt is within bounds
                c_opt = np.clip(c_opt, 1e-12, m)
                
                policy_fine[t, z, i] = c_opt
                
                # Calculate Value V_curr
                m_prime_0 = R * (m - c_opt) + income[0]
                m_prime_1 = R * (m - c_opt) + income[1]
                
                v0 = np.interp(m_prime_0, m_fine, V_next[:, 0])
                v1 = np.interp(m_prime_1, m_fine, V_next[:, 1])
                
                ev = probs[0] * v0 + probs[1] * v1
                V_curr[i, z] = utility(c_opt) + beta * ev
                
                # Calculate Derivative V_prime_curr
                # V'(m) = U'(c) (since d/dc [U(c) + beta E[V'(m') * (-R)]] = 0 at optimum, 
                # so by envelope theorem, dV/dm = U'(c) * dc/dm + 0 ... wait.
                # Bellman: V(m) = max_c { U(c) + beta E[V(R(m-c)+y)] }
                # Envelope Theorem: dV/dm = beta E[ V'(R(m-c)+y) * R ]
                # Also from FOC: U'(c) = beta E[ V'(R(m-c)+y) * R ]
                # So V'(m) = U'(c_opt)
                V_prime_curr[i, z] = utility_prime(c_opt)
                
        V_next = V_curr
        V_prime_next = V_prime_curr
        
    # Interpolate policy from fine grid (500) to output grid (101)
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            # np.interp requires increasing x values. m_fine is increasing.
            # policy_fine[t, z, :] is the optimal consumption on m_fine
            consumption_out[t, z, :] = np.interp(resource_grid, m_fine, policy_fine[t, z, :])
            
    # Ensure terminal condition: c[5] equal to the grid within 1e-9
    # The problem states: "c[5] equal to the grid within 1e-9"
    # This implies that at the last period, consumption must equal resources (m).
    # My backward induction starts at t = horizon-1 (which is index 5 if horizon=6? No, range(horizon-1, -1, -1) is 5,4,3,2,1,0).
    # So t=5 is the first iteration of the loop.
    # At t=5, V_next is 0.
    # FOC: U'(c) - beta * R * V'_6(...) = U'(c) = 0.
    # This has no solution for finite c.
    # The standard assumption is that in the last period, you consume everything.
    # Let's explicitly set t=5 to c=m.
    
    consumption_out[5, :, :] = resource_grid[:, np.newaxis]
    
    return consumption_out
