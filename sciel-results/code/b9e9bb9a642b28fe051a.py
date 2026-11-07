import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Extract parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    n_states = len(income)
    n_out = len(resource_grid)
    
    # Utility and derivative functions (vectorized)
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Internal grid for computation
    # Denser grid for better accuracy as per insights
    n_int = 1000
    m_int = np.geomspace(0.001, 40, n_int)
    log_m_int = np.log(m_int)
    
    # Initialize arrays
    # Policy: shape (horizon, n_states, n_int)
    # Value: shape (horizon, n_states, n_int)
    # Value derivative: shape (horizon, n_states, n_int)
    policy_int = np.zeros((horizon, n_states, n_int))
    value_int = np.zeros((horizon, n_states, n_int))
    val_deriv_int = np.zeros((horizon, n_states, n_int))
    
    # Helper for interpolation on log grid
    # arr: 1D array defined on log_m_int
    # m_next: 1D array of points to interpolate
    def interp_log(arr, m_next):
        m_next = np.maximum(m_next, 1e-12)
        log_m_next = np.log(m_next)
        return np.interp(log_m_next, log_m_int, arr)
    
    # Terminal period t = horizon - 1
    t = horizon - 1
    for z in range(n_states):
        policy_int[t, z, :] = m_int
        value_int[t, z, :] = utility(m_int)
        val_deriv_int[t, z, :] = util_deriv(m_int)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # Check time
            if time.time() > deadline - 5.0:
                # Fallback: use a simple heuristic for remaining periods
                for z_inner in range(n_states):
                    # Use a proportional consumption rule as fallback
                    # c = m / (1 + beta*R) is a rough guess, but for terminal it's m.
                    # For earlier periods, a simple rule is better than nothing.
                    # Let's just use the previous period's policy scaled? 
                    # Or just 0.5*m? 
                    # Actually, if we run out of time, we should return what we have.
                    # But we need to fill the array. 
                    # Let's use the policy from t+1 as a rough estimate for t.
                    if t + 1 < horizon:
                        policy_int[t, z_inner, :] = policy_int[t+1, z_inner, :]
                    else:
                        policy_int[t, z_inner, :] = 0.5 * m_int
                    policy_int[t, z_inner, :] = np.maximum(policy_int[t, z_inner, :], 1e-12)
                    policy_int[t, z_inner, :] = np.minimum(policy_int[t, z_inner, :], m_int)
                    value_int[t, z_inner, :] = utility(policy_int[t, z_inner, :])
                    val_deriv_int[t, z_inner, :] = util_deriv(policy_int[t, z_inner, :])
                # Break out of loops if time is critical
                # We need to break out of the t loop too.
                # Setting a flag or just returning? 
                # The prompt says "return the best result found". 
                # So we should interpolate and return immediately.
                break
            
            # Vectorized Bisection for c*
            # f(c) = u'(c) - beta * R * E[Vp_{t+1}(m')]
            # m' = R*(m-c) + income[z']
            
            m = m_int # (n_int,)
            
            c_low = np.full(n_int, 1e-12)
            c_high = m.copy()
            
            # Iterate bisection
            # 50 iterations should be plenty for double precision on this scale
            for _ in range(50):
                c_mid = (c_low + c_high) / 2.0
                
                # Calculate m_remaining for all grid points
                m_rem = R * (m - c_mid) # (n_int,)
                
                # Calculate expected marginal value E[Vp_{t+1}(m')]
                # m_next[zp] = m_rem + income[zp]
                expected_vp = np.zeros(n_int)
                for zp in range(n_states):
                    m_next = m_rem + income[zp]
                    vp_val = interp_log(val_deriv_int[t+1, zp, :], m_next)
                    expected_vp += transition[z, zp] * vp_val
                
                rhs = beta * R * expected_vp
                lhs = util_deriv(c_mid)
                
                residual = lhs - rhs
                
                # If residual > 0, u'(c) > rhs, so we need to increase c (decrease u')
                # If residual < 0, u'(c) < rhs, so we need to decrease c (increase u')
                
                mask_pos = residual > 0
                c_low[mask_pos] = c_mid[mask_pos]
                
                mask_neg = residual <= 0
                c_high[mask_neg] = c_mid[mask_neg]
            
            c_opt = (c_low + c_high) / 2.0
            c_opt = np.clip(c_opt, 1e-12, m)
            
            policy_int[t, z, :] = c_opt
            
            # Calculate Value
            m_rem = R * (m - c_opt)
            cont = np.zeros(n_int)
            for zp in range(n_states):
                m_next = m_rem + income[zp]
                v_val = interp_log(value_int[t+1, zp, :], m_next)
                cont += transition[z, zp] * v_val
            val_t = utility(c_opt) + beta * cont
            value_int[t, z, :] = val_t
            
            # Calculate Vp using Envelope Theorem
            # Vp_t(m) = beta * R * E[Vp_{t+1}(m')]
            # We already calculated expected_vp in the last bisection step? 
            # No, that was for c_mid. We need it for c_opt.
            # However, c_opt is very close to the last c_mid. 
            # But to be safe and accurate, let's recompute.
            expected_vp_final = np.zeros(n_int)
            for zp in range(n_states):
                m_next = m_rem + income[zp]
                vp_val = interp_log(val_deriv_int[t+1, zp, :], m_next)
                expected_vp_final += transition[z, zp] * vp_val
            vp_t = beta * R * expected_vp_final
            val_deriv_int[t, z, :] = vp_t
            
        # Check time after finishing a period for all states
        if time.time() > deadline - 5.0:
            break

    # Interpolate the internal policies to the output grid
    policy_out = np.zeros((horizon, n_states, n_out))
    
    log_m_out = np.log(resource_grid)
    
    for t in range(horizon):
        for z in range(n_states):
            # Interpolate using log-space
            policy_out[t, z, :] = np.interp(log_m_out, log_m_int, policy_int[t, z, :])
            
            # Enforce constraints: 0 < c <= m + 1e-10
            policy_out[t, z, :] = np.maximum(policy_out[t, z, :], 1e-12)
            policy_out[t, z, :] = np.minimum(policy_out[t, z, :], resource_grid + 1e-10)
            
            # Special case for t = horizon - 1: c should be exactly m
            if t == horizon - 1:
                policy_out[t, z, :] = resource_grid
                
    # Final validity check
    if not np.all(np.isfinite(policy_out)):
        # Replace non-finite values with 0.5 * m
        mask = ~np.isfinite(policy_out)
        policy_out[mask] = 0.5 * resource_grid[np.where(mask)[1], np.where(mask)[2]]
        
    return policy_out
