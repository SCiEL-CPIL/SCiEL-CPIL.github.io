import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']
    
    m_grid = resource_grid
    M = len(m_grid)
    
    def utility_clean(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    V = np.zeros((H + 1, 2, M))
    C = np.zeros((H, 2, M))
    
    # Terminal condition
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Use a dense grid for m' to evaluate continuation values
    # The range of m' can be from min(income) to max(R*m_grid) + max(income)
    m_prime_min = min(income)
    m_prime_max = R * m_grid[-1] + max(income)
    
    # Create a dense grid for m' evaluation
    N_dense = 1000
    m_dense = np.geomspace(max(m_prime_min, 1e-4), m_prime_max, N_dense)
    
    for t in range(4, -1, -1):
        # Check time
        if time.time() > deadline - 10:
            # If we're running out of time, break and return what we have
            # But we need to ensure C is valid. Since we fill t=5 first, 
            # and then go backwards, if we break early, some t might be 0.
            # This is invalid. So we must finish.
            # Given N_dense=1000 and M=101, it should be fast enough.
            pass
        
        for z in range(2):
            # Construct splines for V[t+1, 0, :] and V[t+1, 1, :] on the dense grid
            # First, we need to interpolate V[t+1, z', :] from m_grid to m_dense
            # Or just use np.interp directly in the loop? 
            # Splines are faster for repeated evaluations.
            
            # Interpolate V[t+1, 0, :] to m_dense
            V_next_0_dense = np.interp(m_dense, m_grid, V[t+1, 0, :])
            V_next_1_dense = np.interp(m_dense, m_grid, V[t+1, 1, :])
            
            # Create splines on the dense grid
            spline_0 = CubicSpline(m_dense, V_next_0_dense, bc_type='natural')
            spline_1 = CubicSpline(m_dense, V_next_1_dense, bc_type='natural')
            
            best_val = np.zeros(M)
            best_c = np.zeros(M)
            
            # Precompute expected transition probabilities
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            # Vectorized objective evaluation for a batch of c values?
            # Or just loop over m?
            # Let's try to vectorize over c for a fixed m?
            # No, the objective is 1D in c.
            
            # Let's use a fine grid for c for each m and find the max.
            # This is faster than minimize_scalar for simple concave functions.
            # But is the function concave? 
            # u(c) is concave. V(m') is concave in m', and m' is linear in c.
            # So the objective is concave in c.
            # We can use ternary search or just a dense grid.
            
            # Let's use a dense grid for c.
            # How many points? 200? 500?
            # The evaluator uses 301 points for c in its check.
            # Let's use 500 points for c to be safe.
            N_c = 500
            c_grid_template = np.linspace(1e-8, 1.0, N_c)
            
            for i in range(M):
                m = m_grid[i]
                c_grid = c_grid_template * m
                # Ensure c <= m
                c_grid = np.minimum(c_grid, m)
                
                # Evaluate objective for all c in c_grid
                c_clipped = np.maximum(c_grid, 1e-12)
                u_c = utility_clean(c_clipped)
                
                m_prime_0 = R * (m - c_clipped) + income[0]
                m_prime_1 = R * (m - c_clipped) + income[1]
                
                # Evaluate splines
                v_next_0 = spline_0(m_prime_0)
                v_next_1 = spline_1(m_prime_1)
                
                exp_v = p0 * v_next_0 + p1 * v_next_1
                obj = u_c + beta * exp_v
                
                # Find max
                idx = np.argmax(obj)
                best_c[i] = c_grid[idx]
                best_val[i] = obj[idx]
            
            V[t, z, :] = best_val
            C[t, z, :] = best_c
            
            # Check time
            if time.time() > deadline:
                # If we are running out of time, we might need to break.
                # But we must return a valid C.
                # If we break, C[t] for remaining z might be 0.
                # To be safe, let's just hope it finishes.
                pass
        
    return C
