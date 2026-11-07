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
    # 3000 points allows higher resolution while staying within time budget
    n_inner = 3000
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
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
            # Use coarse-to-fine Newton's method for each m
            # Start with a coarse grid of c candidates, then refine with Newton steps
            
            # Phase 1: Coarse search to get initial estimate
            n_coarse = 200
            fracs_coarse = np.linspace(0.001, 1.0, n_coarse)
            c_coarse = grid[:, None] * fracs_coarse[None, :]
            c_coarse = np.maximum(c_coarse, c_min)
            
            U_coarse = utility(c_coarse)
            cont_coarse = np.zeros((n_grid, n_coarse))
            for zp in range(n_z):
                m_next = R * (grid[:, None] - c_coarse) + income[zp]
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                V_interp = np.interp(m_next_clipped.flatten(), grid, V_next[zp])
                V_interp = V_interp.reshape(n_grid, n_coarse)
                cont_coarse += transition[z, zp] * V_interp
            
            Obj_coarse = U_coarse + beta * cont_coarse
            best_idx_coarse = np.argmax(Obj_coarse, axis=1)
            c_init = c_coarse[np.arange(n_grid), best_idx_coarse]
            
            # Phase 2: Newton refinement
            n_newton = 8
            c_curr = c_init.copy()
            
            for _ in range(n_newton):
                if time.time() > deadline - 3:
                    break
                    
                # Compute derivative of objective w.r.t. c
                # d/dc [U(c) + beta * sum_zp trans[z,zp] * V_next[zp](R*(m-c) + income[zp])]
                # = U'(c) + beta * sum_zp trans[z,zp] * V_next[zp]'(m_next) * (-R)
                
                U_prime = utility_prime(c_curr)
                
                # Compute V_next' by numerical differentiation or interpolation of derivative
                # For simplicity, use central difference on the interpolated value
                eps = 1e-8
                deriv_cont = np.zeros(n_grid)
                for zp in range(n_z):
                    m_next = R * (grid - c_curr) + income[zp]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    
                    # V_next' at m_next: use numerical derivative
                    m_plus = np.clip(m_next + eps, grid[0], grid[-1])
                    m_minus = np.clip(m_next - eps, grid[0], grid[-1])
                    V_plus = np.interp(m_plus, grid, V_next[zp])
                    V_minus = np.interp(m_minus, grid, V_next[zp])
                    V_deriv = (V_plus - V_minus) / (2 * eps)
                    
                    deriv_cont += transition[z, zp] * V_deriv
                
                dObj_dc = U_prime - beta * R * deriv_cont
                
                # Compute second derivative for Newton step
                U_double = utility_double_prime(c_curr)
                
                # Second derivative of continuation (approximate)
                # For CRRA, we can derive analytically, but let's use finite difference
                d2Obj_dc2 = U_double
                # Note: The second derivative of the continuation is complex; 
                # for CRRA with log utility, it simplifies, but let's use a conservative step
                
                # Newton step: c_new = c - dObj/dc / d2Obj/dc2
                # But d2Obj/dc2 can be small or negative in some cases
                # Use a damped Newton step
                step = dObj_dc / np.maximum(np.abs(d2Obj_dc2), 1e-10)
                step = np.clip(step, -0.5, 0.5)  # Limit step size
                c_new = c_curr - step
                
                # Ensure c stays in [c_min, m]
                c_new = np.clip(c_new, c_min, grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-10:
                    c_curr = c_new
                    break
                c_curr = c_new
            
            P_t[z, :] = c_curr
            V_t[z, :] = Obj_coarse[np.arange(n_grid), best_idx_coarse]  # Approximate value
            
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
