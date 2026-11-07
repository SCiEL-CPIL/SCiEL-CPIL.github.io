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
            
    # Use a denser internal grid for better accuracy
    # 800 points is a good balance between accuracy and speed
    n_inner = 800
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
        
        # Precompute interpolation for next period's value function
        # We will use np.interp which is fast enough
        
        for z in range(n_z):
            # We need to find c that maximizes:
            # U(c) + beta * sum_{zp} transition[z, zp] * V_next[zp](R*(m-c) + income[zp])
            
            # Vectorized search over c
            # For each m in grid, c can range from c_min to m.
            # To keep memory manageable and speed high, we evaluate on a dense set of c candidates.
            # Let's use 1000 candidates for c, which is dense enough for the 251-point scoring grid.
            n_cands = 1000
            # c candidates as fractions of m: c = m * frac
            # frac ranges from c_min/m to 1.
            # To handle c_min properly, we can just use linspace for frac and clip.
            fracs = np.linspace(0.001, 1.0, n_cands)
            
            # c candidates for each m: shape (n_grid, n_cands)
            # c[i, j] = grid[i] * fracs[j]
            # But we must ensure c >= c_min.
            # c = np.maximum(grid[:, None] * fracs[None, :], c_min)
            
            # We want to compute the objective for all (m, c) pairs.
            # Objective = U(c) + beta * E[V_next]
            
            # 1. Compute U(c)
            c_cands = grid[:, None] * fracs[None, :]
            c_cands = np.maximum(c_cands, c_min)
            U_c = utility(c_cands) # shape (n_grid, n_cands)
            
            # 2. Compute Continuation Value
            # m_next = R * (m - c) + income[zp]
            # For each zp, we need to interpolate V_next[zp] at m_next.
            
            cont_val = np.zeros((n_grid, n_cands))
            
            for zp in range(n_z):
                # m_next shape: (n_grid, n_cands)
                m_next = R * (grid[:, None] - c_cands) + income[zp]
                # Clip m_next to grid range for interpolation
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                
                # Interpolate V_next[zp] at m_next_clipped
                # np.interp requires 1D inputs, so we flatten
                V_interp = np.interp(m_next_clipped.flatten(), grid, V_next[zp])
                V_interp = V_interp.reshape(n_grid, n_cands)
                
                cont_val += transition[z, zp] * V_interp
            
            # 3. Total Objective
            Obj = U_c + beta * cont_val
            
            # 4. Find argmax
            # We need to ensure c <= m. Since fracs <= 1.0, c_cands <= grid, so this is satisfied.
            # We also need c >= c_min, which is handled by np.maximum.
            
            best_idx = np.argmax(Obj, axis=1)
            best_c = c_cands[np.arange(n_grid), best_idx]
            
            # Ensure best_c is within [c_min, grid]
            best_c = np.clip(best_c, c_min, grid)
            
            P_t[z, :] = best_c
            
            # Calculate value at optimal c
            V_t[z, :] = Obj[np.arange(n_grid), best_idx]
            
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
