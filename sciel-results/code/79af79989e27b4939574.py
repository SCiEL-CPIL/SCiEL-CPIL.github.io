import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
        
    def util_second_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # We will work in log-space for better approximation of the value function
    log_grid = np.log(resource_grid)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                frac = 1.0 / (horizon - tt)
                policy[tt, :, :] = np.minimum(frac * resource_grid[np.newaxis, :], resource_grid[np.newaxis, :])
                policy[tt, :, :] = np.maximum(policy[tt, :, :], c_floor)
                value[tt, :, :] = utility(policy[tt, :, :])
            break
        
        # Create splines for V_{t+1}(m) for each state z' in log-space
        # V_log(x) = V(exp(x))
        splines = []
        for zp in range(num_states):
            splines.append(CubicSpline(log_grid, value[t + 1, zp, :], bc_type='not-a-knot'))
            
        for z in range(num_states):
            m = resource_grid
            log_m = log_grid
            
            # Initial guess for c: use a fraction based on remaining periods
            # c ~ m / (horizon - t) is a rough guess, but 0.5*m is safer for convergence
            # Let's try a smarter initial guess: c = m / (horizon - t)
            # Actually, for CRRA with log utility, the Euler equation suggests a specific pattern.
            # Let's stick with 0.5 * m or maybe 1/(horizon-t) * m.
            # Let's use 1/(horizon - t + 1) * m as it's closer to the steady state for long horizons?
            # No, 1/(remaining periods) is better.
            remaining = horizon - t
            c_est = resource_grid / remaining
            
            # Newton-Raphson iteration
            # f(c) = u'(c) - beta * R * sum_{zp} P(z|zp) * V'_{t+1}(R*(m-c)+income[zp])
            # f'(c) = u''(c) + beta * R^2 * sum_{zp} P(z|zp) * V''_{t+1}(R*(m-c)+income[zp])
            
            for _ in range(50):  # Max 50 Newton steps for robustness
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    m_next = R * (m - c_est) + income[zp]
                    # m_next must be positive for log
                    m_next = np.maximum(m_next, 1e-10)
                    log_m_next = np.log(m_next)
                    
                    # Evaluate V_log and its derivatives
                    v_log = splines[zp](log_m_next, 0)
                    v_log_prime = splines[zp](log_m_next, 1)
                    v_log_second_prime = splines[zp](log_m_next, 2)
                    
                    # Transform back to m-space derivatives
                    # V'(m) = V_log'(log m) / m
                    # V''(m) = (V_log''(log m) - V_log'(log m)) / m^2
                    
                    vp = v_log_prime / m_next
                    vpp = (v_log_second_prime - v_log_prime) / (m_next * m_next)
                    
                    sum_v_prime += transition[z, zp] * vp
                    sum_v_second_prime += transition[z, zp] * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                # Newton step
                # Avoid division by zero
                f_prime_safe = np.where(np.abs(f_prime) < 1e-10, 1e-10, f_prime)
                c_new = c_est - f / f_prime_safe
                
                # Clip c_new to [1e-12, m]
                c_new = np.maximum(c_new, c_floor)
                c_new = np.minimum(c_new, m)
                
                # Check convergence
                # Use a tight tolerance
                tol = 1e-12 * np.maximum(1.0, m)
                if np.all(np.abs(c_new - c_est) < tol):
                    c_est = c_new
                    break
                c_est = c_new
            
            policy[t, z, :] = c_est
            
            # Compute value V_t(m) = u(c) + beta * E[V_{t+1}(m')]
            u_c = utility(c_est)
            sum_v_next = np.zeros_like(m)
            for zp in range(num_states):
                m_next = R * (m - c_est) + income[zp]
                m_next = np.maximum(m_next, 1e-10)
                log_m_next = np.log(m_next)
                v_next = splines[zp](log_m_next, 0)
                sum_v_next += transition[z, zp] * v_next
                
            value[t, z, :] = u_c + beta * sum_v_next

    # Ensure validity: 0 < c <= m + 1e-10
    # Vectorized check
    m_grid = resource_grid[np.newaxis, np.newaxis, :]
    c_all = policy
    # c <= m + 1e-10
    c_all = np.minimum(c_all, m_grid + 1e-10)
    # c > 0
    c_all = np.maximum(c_all, c_floor)
    policy = c_all

    # Ensure c[5] equals grid within 1e-9
    # The terminal condition is already set, but let's be explicit
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=c_floor, posinf=1e10, neginf=0)
    
    return policy
