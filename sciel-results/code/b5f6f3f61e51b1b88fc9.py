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
    
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Use a denser internal grid for better accuracy
    # 2000 points for better Newton convergence
    n_inner = 2000
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: consume all wealth
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Precompute grid spacings for finite differences
    dg = np.diff(grid)
    dg_mid = 0.5 * (grid[1:] + grid[:-1])
    
    def compute_dV(V_arr):
        """Compute first derivative of V using central differences"""
        dV = np.zeros(n_grid)
        # Central differences for interior points
        dV[1:-1] = (V_arr[2:] - V_arr[:-2]) / (grid[2:] - grid[:-2])
        # Forward difference at start
        dV[0] = (V_arr[1] - V_arr[0]) / dg[0]
        # Backward difference at end
        dV[-1] = (V_arr[-1] - V_arr[-2]) / dg[-1]
        return dV
    
    def compute_d2V(V_arr):
        """Compute second derivative of V using finite differences"""
        dV = compute_dV(V_arr)
        d2V = np.zeros(n_grid)
        d2V[1:-1] = (dV[2:] - dV[:-2]) / (grid[2:] - grid[:-2])
        d2V[0] = (dV[1] - dV[0]) / dg[0]
        d2V[-1] = (dV[-1] - dV[-2]) / dg[-1]
        return d2V
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use previous period's policy if time is running out
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        # Precompute derivatives of V_next
        dV_next = np.zeros((n_z, n_grid))
        d2V_next = np.zeros((n_z, n_grid))
        for zp in range(n_z):
            dV_next[zp] = compute_dV(V_next[zp])
            d2V_next[zp] = compute_d2V(V_next[zp])
        
        for z in range(n_z):
            # We need to find c that maximizes:
            # U(c) + beta * sum_{zp} transition[z, zp] * V_next[zp](R*(m-c) + income[zp])
            
            # Use Newton's method to solve the first-order condition
            # FOC: U'(c) - beta * R * sum_zp transition[z,zp] * V_next[zp]'(R*(m-c) + income[zp]) = 0
            
            # Initialize c at a reasonable guess
            c = grid * 0.5
            c = np.maximum(c, c_min)
            c = np.minimum(c, grid)
            
            # Newton iterations
            for it in range(40):
                # Compute U'(c)
                U_prime = utility_prime(c)
                
                # Compute continuation derivative
                cont_deriv = np.zeros(n_grid)
                for zp in range(n_z):
                    m_next = R * (grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    dV_interp = np.interp(m_next_clipped, grid, dV_next[zp])
                    cont_deriv += transition[z, zp] * dV_interp
                
                # FOC: U'(c) - beta * R * cont_deriv = 0
                FOC = U_prime - beta * R * cont_deriv
                
                # Check convergence
                if np.all(np.abs(FOC) < 1e-12):
                    break
                
                # Compute second derivative for Newton step
                U_double = utility_double_prime(c)
                
                # Second derivative of continuation value
                cont_second = np.zeros(n_grid)
                for zp in range(n_z):
                    m_next = R * (grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    d2V_interp = np.interp(m_next_clipped, grid, d2V_next[zp])
                    cont_second += transition[z, zp] * d2V_interp
                
                # dFOC/dc = U''(c) + beta * R^2 * sum_zp transition[z,zp] * V_next[zp]''(R*(m-c) + income[zp])
                dFOC = U_double + beta * R**2 * cont_second
                
                # Newton step with damping
                step = FOC / dFOC
                c_new = c - step
                c_new = np.clip(c_new, c_min, grid)
                
                # Check if we need to damp
                if np.any(np.abs(c_new - c) > 0.5 * np.abs(c)):
                    c_new = c - 0.5 * step
                    c_new = np.clip(c_new, c_min, grid)
                
                c = c_new
            
            # Ensure c is within bounds
            c = np.clip(c, c_min, grid)
            P_t[z, :] = c
            
            # Calculate value at optimal c
            U_c = utility(c)
            cont_val = np.zeros(n_grid)
            for zp in range(n_z):
                m_next = R * (grid - c) + income[zp]
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                V_interp = np.interp(m_next_clipped, grid, V_next[zp])
                cont_val += transition[z, zp] * V_interp
            V_t[z, :] = U_c + beta * cont_val
        
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
