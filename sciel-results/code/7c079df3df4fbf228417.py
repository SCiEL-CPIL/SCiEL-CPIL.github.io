import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Extract parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    m_out = resource_grid
    n_out = len(m_out)
    
    # Internal grids for high-resolution computation
    # Use a dense grid to approximate the continuous optimum
    # 500 points for m, 500 points for s (save fraction)
    m_int = np.geomspace(0.001, 40, 500)
    s_int = np.linspace(0.001, 1.0, 500)
    
    n_m = len(m_int)
    n_s = len(s_int)
    
    # Initialize policy and value arrays
    # Policy: shape (horizon, 2, n_m)
    # Value: shape (horizon, 2, n_m)
    policy_int = np.zeros((horizon, 2, n_m))
    value_int = np.zeros((horizon, 2, n_m))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def utility_derivative(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Backward induction
    # Terminal period t = horizon - 1
    t = horizon - 1
    for z in range(2):
        # At the last period, consume everything
        policy_int[t, z, :] = m_int
        value_int[t, z, :] = utility(m_int)
    
    # Previous periods
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 10.0:
            # Not enough time for full refinement, use current estimates
            break
            
        for z in range(2):
            # Create meshgrid of m and s
            M_grid, S_grid = np.meshgrid(m_int, s_int, indexing='ij')
            C_grid = M_grid * S_grid
            
            # Calculate next period resources m'
            # m' = R * (m - c) + income[z']
            # We need to sum over z' weighted by transition probabilities
            # V_next(z', m')
            
            # Pre-compute expected continuation value for each (m, c)
            # This involves interpolating value_int[t+1, z', :] at m'
            
            cont_val = np.zeros_like(C_grid)
            
            for zp in range(2):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                y_zp = income[zp]
                
                # m' for all (m, c) pairs
                m_prime = R * (M_grid - C_grid) + y_zp
                
                # Interpolate value_int[t+1, zp, :] at m_prime
                v_next = np.interp(m_prime, m_int, value_int[t+1, zp, :])
                
                cont_val += prob * v_next
            
            # Total utility
            u_c = utility(C_grid)
            total_util = u_c + beta * cont_val
            
            # Find the best c for each m
            best_s_idx = np.argmax(total_util, axis=1)
            best_c = C_grid[np.arange(n_m), best_s_idx]
            
            # Store policy and value
            policy_int[t, z, :] = best_c
            value_int[t, z, :] = total_util[np.arange(n_m), best_s_idx]
            
            # High-precision polish: use Euler equation refinement
            # For interior points, the optimal c satisfies:
            # u'(c) = beta * E[u'(c') * R]
            # where c' is next period consumption
            
            # Use Newton's method to refine the policy
            if time.time() > deadline - 5.0:
                continue
                
            c_refined = best_c.copy()
            
            # Iterate Newton refinement
            for newton_iter in range(3):
                # Compute u'(c) at current c
                u_prime_c = utility_derivative(c_refined)
                
                # Compute expected marginal utility of next period consumption
                # We need to estimate c' from the policy at t+1
                # c' = policy(t+1, z', m')
                # m' = R * (m - c) + income[z']
                
                expected_mu_prime = np.zeros(n_m)
                for zp in range(2):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    y_zp = income[zp]
                    
                    m_prime = R * (m_int - c_refined) + y_zp
                    
                    # Get next period policy at m_prime
                    c_prime = np.interp(m_prime, m_int, policy_int[t+1, zp, :])
                    c_prime = np.maximum(c_prime, 1e-12)
                    
                    # u'(c')
                    u_prime_cp = utility_derivative(c_prime)
                    
                    expected_mu_prime += prob * u_prime_cp
                
                # Euler equation: u'(c) = beta * R * E[u'(c')]
                # Residual: f(c) = u'(c) - beta * R * E[u'(c')]
                # We want to find c such that f(c) = 0
                
                # Compute derivative of f w.r.t. c for Newton's method
                # f'(c) = u''(c) - beta * R * d/dc[E[u'(c')]]
                # u''(c) = -u'(c)/c for CRRA
                # For CRRA: u(c) = c^(1-gamma)/(1-gamma), u'(c) = c^(-gamma), u''(c) = -gamma*c^(-gamma-1)
                
                # Simplified: use the fact that for CRRA, the Euler equation has a known structure
                # Let's use a simpler approach: bisection or direct update
                
                # Target: u'(c) = beta * R * E[u'(c')]
                # For CRRA with gamma: c^(-gamma) = beta * R * E[(c')^(-gamma)]
                # So c = (beta * R * E[(c')^(-gamma)])^(-1/gamma)
                
                if np.isclose(gamma, 1.0):
                    # Log utility: u'(c) = 1/c
                    # 1/c = beta * R * E[1/c']
                    # c = 1 / (beta * R * E[1/c'])
                    target_val = beta * R * expected_mu_prime
                    target_val = np.maximum(target_val, 1e-12)
                    c_new = 1.0 / target_val
                else:
                    # CRRA: c^(-gamma) = beta * R * E[(c')^(-gamma)]
                    # c = (beta * R * E[(c')^(-gamma)])^(-1/gamma)
                    target_val = beta * R * expected_mu_prime
                    target_val = np.maximum(target_val, 1e-12)
                    c_new = np.power(target_val, -1.0 / gamma)
                
                # Blend with previous estimate for stability
                c_refined = 0.5 * c_refined + 0.5 * c_new
                
                # Enforce bounds
                c_refined = np.maximum(c_refined, 1e-12)
                c_refined = np.minimum(c_refined, m_int + 1e-10)
            
            # Update policy with refined values
            policy_int[t, z, :] = c_refined
            
            # Recompute value at refined c
            m_prime_refined = np.zeros((2, n_m))
            cont_val_refined = np.zeros(n_m)
            for zp in range(2):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                y_zp = income[zp]
                m_prime_refined[zp, :] = R * (m_int - c_refined) + y_zp
                v_next_refined = np.interp(m_prime_refined[zp, :], m_int, value_int[t+1, zp, :])
                cont_val_refined += prob * v_next_refined
            
            value_int[t, z, :] = utility(c_refined) + beta * cont_val_refined
    
    # Interpolate the internal policies to the output grid
    policy_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            # policy_int[t, z, :] is defined on m_int
            # We want it on m_out
            policy_out[t, z, :] = np.interp(m_out, m_int, policy_int[t, z, :])
            
            # Enforce constraints: 0 < c <= m + 1e-10
            policy_out[t, z, :] = np.maximum(policy_out[t, z, :], 1e-12)
            policy_out[t, z, :] = np.minimum(policy_out[t, z, :], m_out + 1e-10)
            
            # Special case for t = horizon - 1: c should be exactly m
            if t == horizon - 1:
                policy_out[t, z, :] = m_out
    
    return policy_out
