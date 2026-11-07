import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Utility functions
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    def mu(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Marginal utility of consumption
    def mu_consumption(c):
        return mu(c)
    
    # Second derivative of utility (for Newton)
    def d2_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1.0)
    
    # Higher resolution internal grid for better accuracy
    # Use a finer grid for the backward induction
    n_internal = 800
    internal_grid = np.geomspace(0.001, 40.0, n_internal)
    
    # Consumption fraction grid (how much to consume as fraction of wealth)
    n_frac = 601
    frac_grid = np.linspace(0.001, 1.0, n_frac)
    
    # Value function: V[t, z, m] - we'll store as function of m on internal_grid
    # V[t, z, i] where i indexes internal_grid
    # c[t, z, j] where j indexes resource_grid (output grid)
    
    # We'll do backward induction from t = horizon-1 down to 0
    # At t = horizon-1 (last period), the agent consumes everything (or as much as optimal)
    # Actually, at the last period, there's no continuation, so consume all: c = m
    
    # Value at last period: V[horizon-1, z, m] = util(m) (consume all)
    # But we need to be careful: the constraint is c <= m, and m' = R(m-c) + income[z']
    # At the last period, there's no next period, so we just maximize util(c) subject to c <= m
    # Since util is increasing, c = m is optimal at the last period.
    
    # Initialize value at last period
    V = np.zeros((horizon, n_states, n_internal))
    V[horizon-1, :, :] = util(internal_grid)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Break and use what we have
            break
        
        for z in range(n_states):
            # For each m in internal_grid, find optimal c
            # max_c util(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', R*(m-c) + income[z']]
            # subject to 0 < c <= m
            
            # Use vectorized approach with interpolation
            # For each m, we need to evaluate the objective over c values
            # c = m * frac, where frac is in frac_grid
            
            # This is expensive, so let's be smarter
            # For each m, the optimal c satisfies the Euler equation:
            # mu(c) = beta * R * sum_{z'} transition[z, z'] * mu(V_marginal)
            # where V_marginal is the marginal value of wealth
            
            # Actually, let's just do a coarse-to-fine approach
            # First, use a coarse grid, then refine
            
            # For each m in internal_grid, compute the best c
            m_vals = internal_grid
            best_c = np.zeros(n_internal)
            
            # Vectorized: for each m, try all frac values
            # c_matrix[i, j] = m_vals[i] * frac_grid[j]
            # This is n_internal x n_frac which is 800 x 601 = 480800, manageable
            
            # But we need to evaluate the continuation value at m' = R*(m - c) + income[z']
            # for each z', then take the expectation
            
            # Let's do this in chunks to manage memory
            chunk_size = 100
            for start in range(0, n_internal, chunk_size):
                end = min(start + chunk_size, n_internal)
                m_chunk = m_vals[start:end]
                
                # c_matrix: (chunk_size, n_frac)
                c_matrix = np.outer(m_chunk, frac_grid)
                c_matrix = np.maximum(c_matrix, 1e-12)
                
                # For each z', compute m' = R*(m - c) + income[z']
                # m_prime: (chunk_size, n_frac, n_states)
                # This might be too large, so let's do it differently
                
                # Actually, let's compute the objective for each c and take max
                # obj[i, j] = util(c[i,j]) + beta * sum_z' trans[z,z'] * V[t+1, z', interp(m'_ij, internal_grid)]
                
                # To save memory, process one z' at a time
                total_cont = np.zeros((end - start, n_frac))
                for zp in range(n_states):
                    m_prime = R * (m_chunk[:, np.newaxis] - c_matrix) + income[zp]
                    m_prime = np.maximum(m_prime, 0.0)
                    
                    # Interpolate V[t+1, zp, :] onto m_prime
                    # V[t+1, zp, :] is defined on internal_grid
                    V_interp = np.interp(m_prime, internal_grid, V[t+1, zp, :], 
                                         left=V[t+1, zp, 0], right=V[t+1, zp, -1])
                    
                    total_cont += transition[z, zp] * V_interp
                
                # Total objective
                obj = util(c_matrix) + beta * total_cont
                
                # Find best c for each m
                best_idx = np.argmax(obj, axis=1)
                best_c[start:end] = c_matrix[np.arange(end - start), best_idx]
            
            # Store the policy (we'll interpolate to output grid later)
            # For now, store best_c on internal_grid
            # We'll create a policy array policy[t, z, :] on internal_grid
            if not hasattr(V, '_policy'):
                V._policy = np.zeros((horizon, n_states, n_internal))
            V._policy[t, z, :] = best_c
        
        # Check time
        if time.time() > deadline:
            break
    
    # Now interpolate the policy from internal_grid to resource_grid
    # consumption[t, z, resource_index]
    consumption = np.zeros((horizon, n_states, n_grid))
    
    policy = getattr(V, '_policy', None)
    if policy is None:
        # Fallback: if we didn't finish, use a simple rule
        for t in range(horizon):
            for z in range(n_states):
                # Simple rule: consume a fraction based on remaining horizon
                frac = 1.0 / (horizon - t)
                consumption[t, z, :] = np.minimum(frac * resource_grid, resource_grid)
    else:
        for t in range(horizon):
            for z in range(n_states):
                # Interpolate policy from internal_grid to resource_grid
                consumption[t, z, :] = np.interp(resource_grid, internal_grid, policy[t, z, :],
                                                 left=policy[t, z, 0], right=policy[t, z, -1])
                # Ensure 0 < c <= m
                consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Final constraint: c[5] (last period) should equal the grid
    consumption[horizon-1, :, :] = resource_grid[np.newaxis, :, np.newaxis] * np.ones((1, n_states, 1))
    consumption[horizon-1, :, :] = np.clip(consumption[horizon-1, :, :], 1e-12, resource_grid + 1e-10)
    # Actually, at the last period, c = m exactly
    consumption[horizon-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Ensure finite values
    consumption = np.where(np.isfinite(consumption), consumption, 1e-12)
    consumption = np.maximum(consumption, 1e-12)
    
    return consumption
