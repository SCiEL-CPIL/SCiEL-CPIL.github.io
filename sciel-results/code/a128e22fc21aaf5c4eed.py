import numpy as np
from scipy.optimize import minimize_scalar
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Define utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # High resolution grid for internal computation
    # The output must be on resource_grid (101 points)
    # We compute on a finer grid for accuracy, then interpolate or restrict
    # Actually, the task asks for consumption on resource_grid.
    # But the scoring interpolates our policy onto a 251-point grid.
    # So we should compute the policy accurately on a fine grid, 
    # then return the values on the requested resource_grid.
    
    # Let's compute on a fine grid M_fine
    M_fine = np.geomspace(0.001, 40, 2001)
    N_M = len(M_fine)
    
    # Value function V[t, z, m]
    # V is shape (horizon, 2, N_M)
    # Policy C is shape (horizon, 2, N_M)
    
    V = np.zeros((horizon, 2, N_M))
    C = np.zeros((horizon, 2, N_M))
    
    # Terminal condition: c = m
    # V[5, z, m] = u(m)
    for z in range(2):
        V[5, z, :] = u(M_fine)
        C[5, z, :] = M_fine
    
    # Backward induction
    # t from 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break # Return best found so far (which is the terminal if we broke early, but we start from end)
        
        for z in range(2):
            # For each m in M_fine, find optimal c
            # We will vectorize the c search for all m at once?
            # No, c depends on m. 
            # Let's loop over m, but vectorize the c grid.
            
            # To speed up, we can process in batches or just loop.
            # N_M = 2001. Looping 2001 times is fine.
            for i_m, m in enumerate(M_fine):
                # c in [1e-12, m]
                # Use a dense grid for c
                # If m is very small, the grid should be small too?
                # Let's use a fixed number of points, say 200.
                N_c = 200
                c_grid = np.geomspace(1e-12, m, N_c)
                
                # Compute m' for each z'
                # m' = R * (m - c) + income[z']
                # Shape of c_grid: (N_c,)
                # Shape of m_prime: (2, N_c)
                m_prime = R * (m - c_grid)[np.newaxis, :] + income[:, np.newaxis]
                
                # Interpolate V[t+1, z', m']
                # V[t+1, 0, :] is on M_fine
                # We need to interpolate for each z'
                # V_next_0 = np.interp(m_prime[0], M_fine, V[t+1, 0, :])
                # V_next_1 = np.interp(m_prime[1], M_fine, V[t+1, 1, :])
                
                # Vectorized interp
                V_next_0 = np.interp(m_prime[0], M_fine, V[t+1, 0, :])
                V_next_1 = np.interp(m_prime[1], M_fine, V[t+1, 1, :])
                
                # Expected continuation value
                # pi[z', z] is transition[z][z']
                # Note: transition[z] is row z, so transition[z][z'] is prob of going to z'
                # Wait, standard Markov: P(z' | z). 
                # transition = array([[0.85, 0.15], [0.15, 0.85]])
                # transition[0, 0] = 0.85 (z=0 -> z'=0)
                # So expected V = transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1
                
                exp_V = transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1
                
                # Total utility
                total_U = u(c_grid) + beta * exp_V
                
                # Find max
                idx_max = np.argmax(total_U)
                C[t, z, i_m] = c_grid[idx_max]
                V[t, z, i_m] = total_U[idx_max]
                
    # Now we have C on M_fine.
    # We need to return C on resource_grid.
    # resource_grid is 101 points.
    # We can interpolate C[t, z, :] from M_fine to resource_grid.
    
    result = np.zeros((horizon, 2, len(resource_grid)))
    for t in range(horizon):
        for z in range(2):
            result[t, z, :] = np.interp(resource_grid, M_fine, C[t, z, :])
            
    # Enforce constraints
    # 0 < c <= m + 1e-10
    # c[5] == grid
    for t in range(horizon):
        for z in range(2):
            # Clip to [1e-12, resource_grid + 1e-10]
            # But we must ensure c <= m.
            # np.interp might slightly overshoot if not careful, but C[t,z,m] <= m.
            # Interpolation of a monotonic function (which C is) should stay within bounds if endpoints are correct.
            # Let's explicitly clip.
            result[t, z, :] = np.clip(result[t, z, :], 1e-12, resource_grid + 1e-10)
            
            # Ensure c[5] is exactly resource_grid
            if t == 5:
                result[t, z, :] = resource_grid
                
    return result
