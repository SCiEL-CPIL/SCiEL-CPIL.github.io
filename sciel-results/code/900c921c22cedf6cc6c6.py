import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Use a much finer grid for internal calculations to reduce interpolation error
    # The input resource_grid is 101 points. We use 501 points for higher fidelity.
    # We must return a policy on the original resource_grid shape (6, 2, 101).
    # So we solve on the fine grid, then interpolate back to the coarse grid.
    
    n_coarse = len(resource_grid)
    n_fine = 501
    
    # Create a fine grid covering the same range, maybe slightly extended to handle m' > m
    # m' = R(m-c) + income. Max m' approx R*0 + max_income = 1.5 (if m=0) or R*40+1.5 ~ 42.
    # Grid is 0.001 to 40. Let's extend the fine grid slightly to be safe, or clip.
    # Actually, m' can be up to R*40 + 1.5 = 41.5.
    # Let's define the fine grid from min(resource_grid) to max(resource_grid)*1.1 + max(income)
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    
    # We need to handle m' potentially exceeding the grid.
    # The scoring only cares about m in [0.2, 8].
    # For m in [0.2, 8], m' is roughly in [0.5, 10].
    # So the range [0.001, 15] is likely sufficient for high accuracy in the scoring region.
    # However, to be safe for the whole domain, let's go up to 45.
    m_fine_min = 0.0001
    m_fine_max = 45.0
    
    fine_grid = np.geomspace(m_fine_min, m_fine_max, n_fine)
    
    # Policy shape on fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Terminal period: consume everything
    # c = m
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
    
    # We will store interpolators for the next period's policy
    # Using linear interpolation is fast, but PCHIP or cubic might be better.
    # Given the high resolution, linear interpolation on a 501-point grid is very accurate.
    # Let's use linear interpolation for speed, as it's vectorized well in numpy.
    
    p_next_interp = [None] * horizon
    # For terminal period, c(m) = m, so interpolation is identity.
    # We can just use the array directly or an interpolator.
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: linear policy or previous period
            for tt in range(t, horizon - 1):
                policy_fine[tt, :, :] = fine_grid
            break
            
        for z in range(2):
            m_vals = fine_grid
            
            # Initial guess: consume half
            c_lo = np.full(n_fine, 1e-12)
            c_hi = m_vals.copy()
            
            # We want to solve for c such that:
            # u'(c) = beta * R * E_z'[ u'(c_next) ]
            # where c_next = policy[t+1][z'](m_next)
            # m_next = R(m-c) + income[z']
            
            # Since u'(c) is decreasing in c, and E[u'(c_next)] is increasing in c (because m_next decreases as c increases),
            # the LHS decreases and RHS increases.
            # Let f(c) = u'(c) - beta * R * E[u'(c_next)]
            # f(c) is strictly decreasing in c.
            # We want f(c) = 0.
            # f(c_lo) > 0 (consume little, high MU now, low MU future)
            # f(c_hi) < 0 (consume all, low MU now, high MU future)
            
            # Vectorized Bisection
            # We need the next period's policy interpolated at m_next.
            # Let's create interpolators for the next period first.
            # Note: p_next_interp[t+1] should be available.
            
            # Initialize interpolators for t+1
            # We use np.interp which is fast and handles out-of-bounds by clipping.
            
            for iteration in range(40):
                if time.time() > deadline:
                    break
                    
                c_mid = (c_lo + c_hi) / 2.0
                
                # Calculate m_next for both states
                m_next_0 = R * (m_vals - c_mid) + income[0]
                m_next_1 = R * (m_vals - c_mid) + income[1]
                
                # Clip m_next to the fine grid range for interpolation
                # np.interp clamps automatically, but let's be explicit if needed.
                # np.interp(x, xp, fp) clamps to fp[0] and fp[-1] if x is out of bounds.
                
                c_next_0 = np.interp(m_next_0, fine_grid, policy_fine[t + 1, 0, :])
                c_next_1 = np.interp(m_next_1, fine_grid, policy_fine[t + 1, 1, :])
                
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
                
                # Update bounds
                # If residual > 0, u'(c) > target. Since u' is decreasing, we need larger c to lower u'(c).
                # So c is too low. Move lo up.
                # If residual < 0, u'(c) < target. c is too high. Move hi down.
                
                mask_pos = residual > 0
                mask_neg = residual < 0
                
                c_lo[mask_pos] = c_mid[mask_pos]
                c_hi[mask_neg] = c_mid[mask_neg]
                
                # Convergence check
                if np.max(c_hi - c_lo) < 1e-9:
                    break
            
            policy_fine[t, z, :] = (c_lo + c_hi) / 2.0
            
            # Clip to valid range
            policy_fine[t, z, :] = np.clip(policy_fine[t, z, :], 1e-12, m_vals)
            
        # Check time again
        if time.time() > deadline:
             # Fill remaining with terminal-like or linear
             for tt in range(t, horizon - 1):
                 policy_fine[tt, :, :] = fine_grid
             break

    # Interpolate the fine grid policy back to the coarse resource_grid
    policy = np.zeros((horizon, 2, n_coarse))
    
    for t in range(horizon):
        for z in range(2):
            policy[t, z, :] = np.interp(resource_grid, fine_grid, policy_fine[t, z, :])
            
            # Ensure validity: 0 < c <= m
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
            
    # Terminal period must equal grid exactly
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
