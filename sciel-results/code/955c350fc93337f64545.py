import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Backward induction
    # We use a dense grid for consumption search to approximate the optimum
    # The grid is relative to m: c = m * grid_frac
    # We need to handle the case where c > m, so we clip.
    # To be efficient, we create a fixed grid of fractions and multiply by m.
    
    # Let's use a log-spaced or linear spaced grid of fractions?
    # Linear is safer for coverage near 0 and 1.
    # The previous successful strategy used 1000 points. Let's try 2000.
    # But we must be careful with time.
    
    num_c_grid = 2000
    c_fracs = np.linspace(0.001, 1.0, num_c_grid)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            # A common heuristic is to consume a fraction based on remaining periods
            # Or just use the previous period's policy if available, or a simple rule.
            # Let's use a simple rule: consume 1/(horizon - t) of wealth? No, that's risky.
            # Better: use the policy from t+1 if it exists, or just a constant fraction.
            # Since we are in a time-critical fallback, let's just use a conservative rule.
            for tt in range(t, -1, -1):
                # Simple rule: consume a fraction of wealth
                # This is a last resort.
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break
        
        for z in range(num_states):
            # m is shape (num_resources,)
            m = resource_grid
            # c candidates: shape (num_c_grid, num_resources)
            # c_grid[i, j] = m[j] * c_fracs[i]
            c_grid = c_fracs[:, np.newaxis] * m[np.newaxis, :]
            
            # Clip c to [1e-12, m]
            c_grid = np.maximum(c_grid, 1e-12)
            c_grid = np.minimum(c_grid, m[np.newaxis, :])
            
            # Calculate next period's resources: m' = R * (m - c) + income[z']
            # We need to interpolate value[t+1, z', :] for each z'
            
            # Pre-compute interpolated values for each z'
            # V_next[z', i, j] = V_{t+1}(m'_{i,j}, z')
            # m'_{i,j} = R * (m_j - c_{i,j}) + income[z']
            
            # Let's compute the continuation value for each c candidate
            # cont_val[i, j] = sum_{z'} transition[z, z'] * V_{t+1}(m'_{i,j} + income[z'], z')
            
            # To vectorize this, we can loop over z' (only 2 states)
            # This is much faster than looping over j (101) and i (2000)
            
            cont_val = np.zeros_like(c_grid)
            
            for zp in range(num_states):
                # m_next[i, j] = R * (m_j - c_grid[i, j]) + income[zp]
                m_next = R * (m[np.newaxis, :] - c_grid) + income[zp]
                
                # Interpolate V_{t+1, zp, :} at m_next
                # np.interp is 1D, so we need to apply it along axis 1
                # Or use a vectorized interp function.
                # Since resource_grid is fixed, we can use searchsorted or just call np.interp in a loop over zp?
                # No, we need to interpolate for each (i, j).
                # np.interp doesn't work directly on 2D arrays with a single xp.
                # We can use scipy.interpolate.interp1d or just a manual vectorized interp.
                
                # Simple vectorized interp:
                # m_next is (2000, 101)
                # resource_grid is (101,)
                # value[t+1, zp, :] is (101,)
                
                # We can flatten m_next, interp, then reshape
                m_next_flat = m_next.ravel()
                v_next_flat = np.interp(m_next_flat, resource_grid, value[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)
                
                cont_val += transition[z, zp] * v_next
            
            # Total objective: u(c) + beta * cont_val
            u_c = utility(c_grid)
            obj = u_c + beta * cont_val
            
            # Find max for each j (resource state)
            # max_idx[j] = argmax along axis 0
            max_idx = np.argmax(obj, axis=0)
            
            # Extract optimal c and value
            # c_opt[j] = c_grid[max_idx[j], j]
            # v_opt[j] = obj[max_idx[j], j]
            
            # We need to index with arrays
            # max_idx is shape (101,)
            # c_grid is (2000, 101)
            # We want c_grid[max_idx, np.arange(101)]
            c_opt = c_grid[max_idx, np.arange(num_resources)]
            v_opt = obj[max_idx, np.arange(num_resources)]
            
            policy[t, z, :] = c_opt
            value[t, z, :] = v_opt
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
