import numpy as np
import time
from scipy.optimize import brentq
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
    
    n_out = len(resource_grid)
    
    # Use a fine internal grid for high accuracy
    n_fine = 801
    m_fine_min = 0.0001
    m_fine_max = 45.0
    fine_grid = np.geomspace(m_fine_min, m_fine_max, n_fine)
    
    # Policy shape on fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Terminal period: consume everything
    policy_fine[horizon - 1, :, :] = fine_grid
    
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Store PCHIP interpolators for the next period's policy
    # We'll build them as we go backward
    
    # For the terminal period, c(m) = m, so we can use a simple interpolator
    # But let's just use np.interp for terminal since it's identity
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with terminal-like policy
            for tt in range(t, horizon - 1):
                policy_fine[tt, :, :] = fine_grid
            break
            
        # Build PCHIP interpolators for the next period's policy
        interp_next = [None] * 2
        for z in range(2):
            # Ensure policy is strictly increasing for PCHIP
            # c(m) should be increasing in m
            c_vals = policy_fine[t + 1, z, :]
            # Make sure it's monotonic
            c_vals = np.maximum.accumulate(c_vals)
            interp_next[z] = PchipInterpolator(fine_grid, c_vals, extrapolate=False)
        
        for z in range(2):
            m_vals = fine_grid
            
            # Initial guess: use a simple heuristic
            # For CRRA, c is roughly proportional to m
            # Let's start with c = 0.5 * m
            c_lo = np.full(n_fine, 1e-12)
            c_hi = m_vals.copy()
            
            # Vectorized Newton-Raphson with bisection fallback
            # We want to solve: u'(c) = beta * R * E[u'(c_next)]
            # Let f(c) = u'(c) - beta * R * E[u'(c_next)]
            # f'(c) = u''(c) + beta * R * E[u''(c_next) * (-R)]  (since dm/dc = -R)
            #       = -gamma * c^(-gamma-1) + beta * R^2 * E[gamma * c_next^(-gamma-1)]
            
            for iteration in range(50):
                if time.time() > deadline:
                    break
                    
                c_mid = (c_lo + c_hi) / 2.0
                
                # Calculate m_next for both states
                m_next_0 = R * (m_vals - c_mid) + income[0]
                m_next_1 = R * (m_vals - c_mid) + income[1]
                
                # Clip m_next to the fine grid range
                m_next_0 = np.clip(m_next_0, fine_grid[0], fine_grid[-1])
                m_next_1 = np.clip(m_next_1, fine_grid[0], fine_grid[-1])
                
                # Evaluate next period's policy
                c_next_0 = interp_next[0](m_next_0)
                c_next_1 = interp_next[1](m_next_1)
                
                # Ensure c_next is valid
                c_next_0 = np.maximum(c_next_0, 1e-12)
                c_next_1 = np.maximum(c_next_1, 1e-12)
                
                # Marginal utilities
                mu_c = marginal_utility(c_mid)
                mu_next_0 = marginal_utility(c_next_0)
                mu_next_1 = marginal_utility(c_next_1)
                
                E_mu_next = transition[z, 0] * mu_next_0 + transition[z, 1] * mu_next_1
                target = beta * R * E_mu_next
                
                residual = mu_c - target
                
                # Derivative of residual w.r.t. c
                # f'(c) = u''(c) - beta * R * d/dc[E[u'(c_next)]]
                # d/dc[E[u'(c_next)]] = E[u''(c_next) * dm_next/dc]
                # dm_next/dc = -R
                # So f'(c) = -gamma * c^(-gamma-1) - beta * R * E[-gamma * c_next^(-gamma-1) * (-R)]
                #          = -gamma * c^(-gamma-1) - beta * R^2 * gamma * E[c_next^(-gamma-1)]
                
                if abs(gamma - 1.0) < 1e-9:
                    second_mu_c = -1.0 / c_mid**2
                    second_mu_next_0 = -1.0 / c_next_0**2
                    second_mu_next_1 = -1.0 / c_next_1**2
                else:
                    second_mu_c = -gamma * c_mid ** (-gamma - 1)
                    second_mu_next_0 = -gamma * c_next_0 ** (-gamma - 1)
                    second_mu_next_1 = -gamma * c_next_1 ** (-gamma - 1)
                
                E_second_mu_next = transition[z, 0] * second_mu_next_0 + transition[z, 1] * second_mu_next_1
                derivative = second_mu_c - beta * R * R * E_second_mu_next
                
                # Newton step
                c_newton = c_mid - residual / derivative
                
                # Check if Newton step is within bounds
                valid_newton = (c_newton > c_lo) & (c_newton < c_hi)
                
                # Update bounds based on residual sign
                mask_pos = residual > 0
                mask_neg = residual < 0
                
                c_lo[mask_pos] = c_mid[mask_pos]
                c_hi[mask_neg] = c_mid[mask_neg]
                
                # Use Newton step where valid, otherwise use midpoint
                c_new = np.where(valid_newton, c_newton, (c_lo + c_hi) / 2.0)
                
                # Ensure c_new is within bounds
                c_new = np.clip(c_new, c_lo, c_hi)
                
                # Check convergence
                if np.max(np.abs(c_new - c_mid)) < 1e-10:
                    c_mid = c_new
                    break
                
                c_mid = c_new
            
            policy_fine[t, z, :] = c_mid
            
            # Clip to valid range
            policy_fine[t, z, :] = np.clip(policy_fine[t, z, :], 1e-12, m_vals)
            
            # Ensure monotonicity
            policy_fine[t, z, :] = np.maximum.accumulate(policy_fine[t, z, :])
    
    # Interpolate the fine grid policy back to the coarse resource_grid
    policy = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            # Use PCHIP for smoother interpolation
            c_vals = policy_fine[t, z, :]
            c_vals = np.maximum.accumulate(c_vals)
            interp = PchipInterpolator(fine_grid, c_vals, extrapolate=False)
            policy[t, z, :] = interp(resource_grid)
            
            # Ensure validity: 0 < c <= m
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Terminal period must equal grid exactly
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
