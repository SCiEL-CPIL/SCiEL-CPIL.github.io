"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid  # shape (101,)
    n_m = len(m_grid)
    n_z = 2  # number of income states
    
    # CRRA utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Backward induction
    # V[t, z, m] = value at time t, state z, resources m
    # c[t, z, m] = optimal consumption
    
    # Terminal condition: at t = horizon-1 (last period), consume everything
    c = np.zeros((horizon, n_z, n_m))
    V = np.zeros((horizon + 1, n_z, n_m))
    
    # At the final period, consumption equals resources (no future)
    c[horizon - 1, :, :] = m_grid[None, :]
    
    # For t from horizon-2 down to 0
    # We need to solve the Bellman equation
    # V_t(z, m) = max_c { u(c) + beta * sum_z' P(z'|z) V_{t+1}(z', m') }
    # where m' = R(m - c) + income[z']
    
    # Use vectorized grid search over consumption
    # For each (t, z, m), search over c in [0, m]
    
    # Number of consumption candidates - use 1500 for fine resolution
    n_cand = 1500  # fine grid
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 3:
            # Safety: fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                c[tt, :, :] = m_grid[None, :]
            break
        
        for z in range(n_z):
            # For each m in the grid, find optimal c
            # m_grid has shape (n_m,)
            # We want to compute for all m simultaneously
            
            # Candidate consumptions: c = m * linspace(1e-4, 1, n_cand)
            # Shape: (n_m, n_cand)
            fracs = np.linspace(1e-4, 1.0, n_cand)  # (n_cand,)
            cands = m_grid[:, None] * fracs[None, :]  # (n_m, n_cand)
            
            # Ensure c <= m
            cands = np.minimum(cands, m_grid[:, None])
            
            # Compute utility for each candidate
            u_vals = utility(cands)  # (n_m, n_cand)
            
            # Compute next period resources for each z'
            # m' = R * (m - c) + income[z']
            # For each z', shape: (n_m, n_cand)
            continuation = np.zeros((n_m, n_cand))
            for zp in range(n_z):
                m_next = R * (m_grid[:, None] - cands) + income[zp]
                # Clip to grid range for interpolation
                m_next = np.clip(m_next, m_grid[0], m_grid[-1])
                # Interpolate V[t+1, zp, :] onto m_next
                V_interp = np.interp(m_next, m_grid, V[t + 1, zp, :])
                continuation += transition[z, zp] * V_interp
            
            # Total objective
            obj = u_vals + beta * continuation  # (n_m, n_cand)
            
            # Find argmax over candidates
            best_idx = np.argmax(obj, axis=1)  # (n_m,)
            
            # Extract optimal consumption
            c_opt = cands[np.arange(n_m), best_idx]  # (n_m,)
            
            # Store
            c[t, z, :] = c_opt
            V[t, z, :] = obj[np.arange(n_m), best_idx]
        
        # Update V for terminal check (not strictly needed but for consistency)
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c[t, z, :] = np.maximum(c[t, z, :], 1e-12)
            c[t, z, :] = np.minimum(c[t, z, :], m_grid + 1e-10)
    
    # Terminal period: c[5] should equal the grid
    c[horizon - 1, :, :] = m_grid[None, :]
    
    return c
# EVOLVE-BLOCK-END
