import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal condition: c_H(m) = m
    # For all z, c[T-1, z, :] = resource_grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Pre-compute grid properties for interpolation
    # np.interp requires increasing xp. resource_grid is increasing.
    
    def inv_marginal_utility(mu):
        """
        Inverse of marginal utility.
        mu = u'(c)
        Returns c such that u'(c) = mu.
        """
        mu = np.maximum(mu, 1e-15) # Avoid division by zero or negative issues
        if gamma == 1:
            # u'(c) = 1/c => c = 1/mu
            return 1.0 / mu
        else:
            # u'(c) = c^(-gamma) => c = mu^(-1/gamma)
            return mu ** (-1.0 / gamma)
    
    def marginal_utility(c):
        """
        Marginal utility u'(c).
        """
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Work backwards from t = horizon-2 to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
            
        for z in range(Z):
            # Initialize guess for consumption at time t, state z
            # A good guess is the average of the next period's consumption
            # or simply half of the resources
            C = resource_grid.copy() / 2.0
            
            # Fixed point iteration
            max_iter = 100
            tol = 1e-10
            
            for _ in range(max_iter):
                C_old = C.copy()
                
                # Calculate next period resources m' for each z'
                # m' = R * (m - c) + income[z']
                # m is resource_grid (shape M,)
                # C is current consumption guess (shape M,)
                # We need m' for z'=0 and z'=1
                
                m_prime_0 = R * (resource_grid - C) + income[0]
                m_prime_1 = R * (resource_grid - C) + income[1]
                
                # Clip m' to the grid range for interpolation
                m_min = resource_grid[0]
                m_max = resource_grid[-1]
                
                m_prime_0_clipped = np.clip(m_prime_0, m_min, m_max)
                m_prime_1_clipped = np.clip(m_prime_1, m_min, m_max)
                
                # Interpolate next period consumption policies
                # consumption[t+1, z', :] is the policy for next period
                c_next_0 = np.interp(m_prime_0_clipped, resource_grid, consumption[t + 1, 0, :])
                c_next_1 = np.interp(m_prime_1_clipped, resource_grid, consumption[t + 1, 1, :])
                
                # Calculate marginal utility of next period consumption
                mu_next_0 = marginal_utility(c_next_0)
                mu_next_1 = marginal_utility(c_next_1)
                
                # Calculate expected marginal utility
                # E[mu'] = P(z'=0|z) * mu_0 + P(z'=1|z) * mu_1
                expected_mu = transition[z, 0] * mu_next_0 + transition[z, 1] * mu_next_1
                
                # Euler equation: u'(c) = beta * R * E[mu']
                # c = inv_u'(beta * R * E[mu'])
                target_mu = beta * R * expected_mu
                C_new = inv_marginal_utility(target_mu)
                
                # Clip consumption to valid range (0, m]
                # c must be > 0 and <= m
                C_new = np.clip(C_new, 1e-12, resource_grid)
                
                # Check convergence
                if np.max(np.abs(C_new - C_old)) < tol:
                    C = C_new
                    break
                else:
                    C = C_new
            
            consumption[t, z, :] = C
    
    # Final validation and cleanup
    # Ensure c[5] equals grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    # We already clipped to resource_grid, which is m.
    # But let's be safe.
    for t in range(horizon):
        for z in range(Z):
            c_vec = consumption[t, z, :]
            # c must be > 0
            c_vec[c_vec <= 0] = 1e-12
            # c must be <= m
            # Since we clipped to resource_grid, c <= m is satisfied.
            # However, due to floating point, it might slightly exceed?
            # np.clip ensures <= resource_grid.
            consumption[t, z, :] = c_vec
            
    return consumption
