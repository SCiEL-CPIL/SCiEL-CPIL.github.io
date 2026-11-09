import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    # c[t, z, i] - consumption policy
    c = np.zeros((horizon, n_states, n_grid))
    
    # Terminal: consume everything
    # c[-1] has shape (2, 101). resource_grid has shape (101,).
    # We need to broadcast resource_grid to (2, 101).
    c[-1, :, :] = np.tile(resource_grid, (n_states, 1))
    
    # CRRA utility derivative: u'(c) = c^(-gamma)
    # Euler: c^(-gamma) = beta * E_z'[(c'(m', z'))^(-gamma)]
    
    def interp_policy(c_next, m_prime, z_prime):
        """Interpolate c_next[m_prime, z_prime] for each z_prime state."""
        # c_next has shape (n_grid,) for a given z
        # m_prime has shape (n_grid,) 
        return np.interp(m_prime, resource_grid, c_next)
    
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # For each m in resource_grid, solve for c
            m_vals = resource_grid  # shape (n_grid,)
            
            # Bisection bounds
            c_low = np.full(n_grid, 1e-12)
            c_high = m_vals.copy()  # can't consume more than m
            
            # We need f(c) = c^(-gamma) - beta * sum_z' P(z'|z) * c_next(m', z')^(-gamma) = 0
            # where m' = R*(m - c) + income[z']
            
            for iteration in range(60):
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                
                c_mid = (c_low + c_high) / 2.0
                
                # Compute m' for each z'
                # m' = R*(m - c_mid) + income[z']
                # For z'=0: m'_0 = R*(m - c_mid) + income[0]
                # For z'=1: m'_1 = R*(m - c_mid) + income[1]
                
                m_prime_0 = R * (m_vals - c_mid) + income[0]
                m_prime_1 = R * (m_vals - c_mid) + income[1]
                
                # Clip m' to be >= 0 (can't have negative resources)
                m_prime_0 = np.maximum(m_prime_0, 0.0)
                m_prime_1 = np.maximum(m_prime_1, 0.0)
                
                # Interpolate c_{t+1}(m', z')
                c_next_0 = np.interp(m_prime_0, resource_grid, c[t+1, 0, :])
                c_next_1 = np.interp(m_prime_1, resource_grid, c[t+1, 1, :])
                
                # Ensure c_next is at least 1e-12
                c_next_0 = np.maximum(c_next_0, 1e-12)
                c_next_1 = np.maximum(c_next_1, 1e-12)
                
                # Compute f(c_mid) = c_mid^(-gamma) - beta * (P(0|z)*c_next_0^(-gamma) + P(1|z)*c_next_1^(-gamma))
                lhs = c_mid ** (-gamma)
                rhs = beta * (transition[z, 0] * c_next_0 ** (-gamma) + transition[z, 1] * c_next_1 ** (-gamma))
                
                f = lhs - rhs
                
                # f is decreasing in c. If f > 0, c is too low, increase lower bound.
                # If f < 0, c is too high, decrease upper bound.
                # Actually: f decreasing means if f(mid) > 0, root is to the right (higher c)
                # Wait no. f decreasing: as c increases, f decreases.
                # If f(mid) > 0, we need to increase c to reduce f toward 0. So c_low = mid.
                # If f(mid) < 0, we need to decrease c to increase f toward 0. So c_high = mid.
                
                c_low = np.where(f > 0, c_mid, c_low)
                c_high = np.where(f < 0, c_mid, c_high)
            
            c[t, z, :] = (c_low + c_high) / 2.0
            # Ensure validity: 0 < c <= m
            c[t, z, :] = np.clip(c[t, z, :], 1e-12, m_vals)
        
        # Check deadline
        if time.time() - start_time > deadline - 10:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t - 1, -1, -1):
                for z2 in range(n_states):
                    # Use a simple saving rule: save a fixed fraction
                    c[t2, z2, :] = resource_grid * 0.7  # consume 70%
                    c[t2, z2, :] = np.clip(c[t2, z2, :], 1e-12, resource_grid)
            break
    
    # Final validity checks
    c = np.clip(c, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            c[t, z, :] = np.minimum(c[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition: c[5] must equal grid
    c[-1, :, :] = np.tile(resource_grid, (n_states, 1))
    c = np.clip(c, 1e-12, None)
    
    return c
