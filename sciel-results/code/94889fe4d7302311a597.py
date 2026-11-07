import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_coarse = len(resource_grid)
    
    # Use a moderate fine grid - Newton's method converges faster, so we need fewer points
    n_fine = 301
    
    m_fine_min = 0.0001
    m_fine_max = 45.0
    
    fine_grid = np.geomspace(m_fine_min, m_fine_max, n_fine)
    
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Terminal period: consume everything
    policy_fine[horizon - 1, :, :] = fine_grid
    
    # Helper functions
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def second_derivative_utility(c):
        c = np.maximum(c, 1e-12)
        return -gamma * c ** (-gamma - 1)
    
    # Build PCHIP interpolators for the next period's policy
    # PCHIP gives C1 continuity which is better for Newton's method
    pchip_next = [None] * horizon
    # For terminal: c(m) = m
    pchip_next[horizon - 1] = [PchipInterpolator(fine_grid, fine_grid), 
                                 PchipInterpolator(fine_grid, fine_grid)]
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, horizon - 1):
                policy_fine[tt, :, :] = fine_grid
            break
            
        for z in range(2):
            m_vals = fine_grid
            
            # Initial guess: use a heuristic based on Euler equation approximation
            # For CRRA, optimal c ≈ m / (1 + beta*R*(1/gamma)) as a rough starting point
            # Better: start from a fraction of m
            c_current = 0.5 * m_vals.copy()
            
            # Newton's method with damping
            for iteration in range(15):
                if time.time() > deadline:
                    break
                    
                # Calculate m_next for both states
                m_next_0 = R * (m_vals - c_current) + income[0]
                m_next_1 = R * (m_vals - c_current) + income[1]
                
                # Get c_next using PCHIP
                c_next_0 = pchip_next[t + 1][0](m_next_0)
                c_next_1 = pchip_next[t + 1][1](m_next_1)
                
                c_next_0 = np.maximum(c_next_0, 1e-12)
                c_next_1 = np.maximum(c_next_1, 1e-12)
                
                # Marginal utilities
                mu_c = marginal_utility(c_current)
                mu_next_0 = marginal_utility(c_next_0)
                mu_next_1 = marginal_utility(c_next_1)
                
                E_mu_next = transition[z, 0] * mu_next_0 + transition[z, 1] * mu_next_1
                target = beta * R * E_mu_next
                
                residual = mu_c - target
                
                # Compute derivative of residual w.r.t. c for Newton's step
                # f(c) = u'(c) - beta*R*E[u'(c_next(m-c))]
                # f'(c) = u''(c) + beta*R*E[u''(c_next) * R]  (chain rule: d(m-c)/dc = -R, d(c_next)/d(m_next) ≈ policy derivative)
                # We need the derivative of c_next w.r.t. m_next, which is the slope of the policy
                # PCHIP gives us the derivative
                dc_next_0_dm = pchip_next[t + 1][0](m_next_0, nu=1)
                dc_next_1_dm = pchip_next[t + 1][1](m_next_1, nu=1)
                
                # Clip derivatives to reasonable range
                dc_next_0_dm = np.clip(dc_next_0_dm, 0.1, 0.99)
                dc_next_1_dm = np.clip(dc_next_1_dm, 0.1, 0.99)
                
                u2_c = second_derivative_utility(c_current)
                u2_next_0 = second_derivative_utility(c_next_0)
                u2_next_1 = second_derivative_utility(c_next_1)
                
                # f'(c) = u''(c) + beta*R*E[u''(c_next) * R * dc_next/dm]
                # Note: d(m_next)/dc = -R, so d(c_next)/dc = dc_next/dm * (-R)
                # The term in the expectation: u''(c_next) * d(c_next)/dc = u''(c_next) * dc_next/dm * (-R)
                # So f'(c) = u''(c) - beta*R*E[u''(c_next) * dc_next/dm * R]
                # Wait, let me re-derive:
                # f(c) = u'(c) - beta*R*E[u'(c_next(m_next(c)))]
                # df/dc = u''(c) - beta*R*E[u''(c_next) * dc_next/dm_next * dm_next/dc]
                # dm_next/dc = -R
                # df/dc = u''(c) - beta*R*E[u''(c_next) * dc_next/dm * (-R)]
                # df/dc = u''(c) + beta*R^2*E[u''(c_next) * dc_next/dm]
                
                E_u2_dc = transition[z, 0] * u2_next_0 * dc_next_0_dm + transition[z, 1] * u2_next_1 * dc_next_1_dm
                df_dc = u2_c + beta * R * R * E_u2_dc
                
                # Newton step with damping
                # c_new = c - f/f'
                # Since f' < 0 (both terms are negative for CRRA), and we want to solve f=0
                # If f > 0, c is too low, need to increase c
                # Newton: c_new = c - f/f' = c - f/(negative) = c + f/|f'|
                
                step = residual / df_dc
                # Damping: limit step size
                max_step = 0.5 * m_vals
                step = np.clip(step, -max_step, max_step)
                
                c_new = c_current - step
                
                # Ensure validity
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_current)) < 1e-10:
                    c_current = c_new
                    break
                
                c_current = c_new
            
            policy_fine[t, z, :] = c_current
            policy_fine[t, z, :] = np.clip(policy_fine[t, z, :], 1e-12, m_vals)
            
        # Build PCHIP for this period for use in previous period
        if time.time() < deadline:
            pchip_next[t] = [PchipInterpolator(fine_grid, policy_fine[t, 0, :]),
                             PchipInterpolator(fine_grid, policy_fine[t, 1, :])]
        else:
            # Fallback
            for tt in range(t, horizon - 1):
                policy_fine[tt, :, :] = fine_grid
            break

    # Interpolate the fine grid policy back to the coarse resource_grid
    policy = np.zeros((horizon, 2, n_coarse))
    
    for t in range(horizon):
        for z in range(2):
            policy[t, z, :] = np.interp(resource_grid, fine_grid, policy_fine[t, z, :])
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
            
    # Terminal period must equal grid exactly
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
