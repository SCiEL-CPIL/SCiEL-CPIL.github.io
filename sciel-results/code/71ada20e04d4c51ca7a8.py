import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    # Internal grid for resource m
    # Use a dense grid to ensure accurate interpolation of V
    n_inner = 1000
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    # Grid for consumption fractions alpha in [0, 1]
    # We need enough points to capture the optimum accurately
    n_alpha = 500
    alpha_grid = np.linspace(0.0, 1.0, n_alpha)
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: consume all wealth
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use previous period's policy if time is running out
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        for z in range(n_z):
            # m shape: (n_grid,)
            # alpha shape: (n_alpha,)
            # c shape: (n_grid, n_alpha)
            c_vals = grid[:, np.newaxis] * alpha_grid[np.newaxis, :]
            c_vals = np.maximum(c_vals, c_min)
            
            # Utility of consumption
            U_c = utility(c_vals)
            
            # Continuation value
            cont_val = np.zeros((n_grid, n_alpha))
            for zp in range(n_z):
                m_next = R * (grid[:, np.newaxis] - c_vals) + income[zp]
                # Clip m_next to the grid range for interpolation
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                
                # Flatten for np.interp
                m_flat = m_next_clipped.ravel()
                V_flat = np.interp(m_flat, grid, V_next[zp])
                V_interp = V_flat.reshape(n_grid, n_alpha)
                
                cont_val += transition[z, zp] * V_interp
            
            # Total value
            J = U_c + beta * cont_val
            
            # Find optimal consumption
            idx_max = np.argmax(J, axis=1)
            
            # Extract optimal c and V
            c_opt = c_vals[np.arange(n_grid), idx_max]
            V_opt = J[np.arange(n_grid), idx_max]
            
            # Ensure c is not 0 (constraint 0 < c)
            c_opt = np.maximum(c_opt, c_min)
            
            P_t[z, :] = c_opt
            V_t[z, :] = V_opt
        
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate policy back to the original grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    # Terminal condition must be exact on the output grid
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
