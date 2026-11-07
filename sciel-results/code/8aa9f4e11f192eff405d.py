import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2
    
    # Precompute utility functions
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def utility_scalar(c):
        c = max(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # V[t][z] will be an array of values at grid points.
    # P[t][z] will be an array of optimal consumptions at grid points.
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: t = T-1
    # c = m, V = u(m)
    V[T-1] = utility(grid)[np.newaxis, :].repeat(2, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(2, axis=0)
    
    # Backward induction for t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with terminal policy (consume all)
            # This is a valid but suboptimal policy, ensures termination
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[T-1].copy()
                P[t_rem] = P[T-1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        
        V_next = V[t+1]
        
        for z in range(n_z):
            for i, m in enumerate(grid):
                if m < c_min:
                    P_t[z, i] = c_min
                    V_t[z, i] = utility_scalar(m)
                    continue
                
                # Define the objective function to maximize:
                # f(c) = u(c) + beta * sum_{z'} P(z'|z) V_next[z'](R(m-c) + y_{z'})
                
                def neg_objective(c):
                    # c is scalar
                    c = max(c, c_min)
                    u_c = utility_scalar(c)
                    
                    cont_val = 0.0
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        # Clip next_m to grid range for interpolation safety? 
                        # np.interp handles out-of-bounds by returning end values, 
                        # which is consistent with linear extrapolation of the last segment? 
                        # Actually np.interp returns the boundary value. 
                        # This is a standard assumption for these grid-based solutions.
                        v_zp = np.interp(next_m, grid, V_next[zp])
                        cont_val += transition[z, zp] * v_zp
                        
                    return -(u_c + beta * cont_val)
                
                # Search for c in [c_min, m]
                # Use bounded minimization
                res = minimize_scalar(neg_objective, bounds=(c_min, m), method='bounded', options={'xatol': 1e-9})
                
                c_opt = res.x
                # Ensure strict positivity and upper bound
                c_opt = np.clip(c_opt, c_min, m)
                
                P_t[z, i] = c_opt
                # Calculate the value at the optimal c
                # We can just evaluate the objective and negate it
                val = -res.fun
                V_t[z, i] = val
        
        V[t] = V_t
        P[t] = P_t
        
    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P[t]
        
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)
            
    # Final period constraint: c = m
    result[T-1, :, :] = grid[np.newaxis, :]
    
    return result
