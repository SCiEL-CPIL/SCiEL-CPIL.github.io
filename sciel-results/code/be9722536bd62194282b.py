"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Continuation value: V[t][z, m]
    # At t = horizon-1 (last period), consumption = m (consume all)
    # We need c[t, z, m_index]
    
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Last period: consume all
    c_policy[-1] = m_grid[np.newaxis, :]
    
    # Backward induction
    # V_next[z, m] = expected value of next period's consumption utility + continuation
    # Actually, we need to solve: at each t, given m and z, choose c to maximize:
    #   beta * utility(c) + beta * E_z'[ V_{t+1}(z', m') ]
    # where m' = R*(m - c) + income[z']
    
    # For the last period (t = horizon-1), c = m (consume all)
    # For earlier periods, we need to optimize c.
    
    # V[t][z, m] represents the value from period t onwards given state (z, m)
    # V[horizon-1][z, m] = utility(m)  [since c = m]
    
    V = np.zeros((horizon, n_z, n_m))
    V[-1] = utility(m_grid)[np.newaxis, :]
    
    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        V_t = np.zeros((n_z, n_m))
        for z in range(n_z):
            # For each m, find optimal c
            for i_m in range(n_m):
                m = m_grid[i_m]
                # c must satisfy 0 < c <= m
                # m' = R*(m - c) + income[z']
                # We want to maximize: beta * utility(c) + beta * sum_z' trans[z,z'] * V[t+1, z', m']
                
                # Use a grid of c values: from small fraction to m
                # c_grid = m * linspace(0.001, 1, budget)
                c_grid = m * np.linspace(0.001, 1.0, budget)
                
                # Compute m' for each z'
                # m'_z' = R*(m - c) + income[z']
                # This is a 2D array: [n_z', budget]
                m_prime = R * (m - c_grid)[np.newaxis, :] + income[:, np.newaxis]
                
                # Interpolate V[t+1] at m_prime for each z'
                # V[t+1, z', :] is defined on m_grid
                V_next_z = V[t + 1]  # shape (n_z, n_m)
                
                # For each z', interpolate V_next_z[z', :] at m_prime[z', :]
                V_interp = np.zeros((n_z, budget))
                for zp in range(n_z):
                    # Clip m_prime to grid range
                    m_prime_clipped = np.clip(m_prime[zp], m_grid[0], m_grid[-1])
                    V_interp[zp, :] = np.interp(m_prime_clipped, m_grid, V_next_z[zp, :])
                
                # Expected continuation: sum over z' of transition[z, z'] * V_interp[z', :]
                cont = np.sum(transition[z, :, np.newaxis] * V_interp, axis=0)
                
                # Objective: beta * utility(c) + beta * cont
                obj = beta * utility(c_grid) + beta * cont
                
                # Find argmax
                best_idx = np.argmax(obj)
                c_policy[t, z, i_m] = c_grid[best_idx]
                V_t[z, i_m] = obj[best_idx]
        
        V[t] = V_t
    
    # Ensure c[5] (last period) equals grid
    c_policy[-1] = m_grid[np.newaxis, :]
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, m_grid + 1e-10)
    
    return c_policy
# EVOLVE-BLOCK-END
