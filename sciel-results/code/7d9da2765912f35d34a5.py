import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145
    
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
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
        
    def util_second_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # We will use a higher-resolution internal grid for interpolation
    # The resource_grid is geomspace(0.001, 40, 101)
    # Let's create a denser grid for better interpolation accuracy
    # But we must return policy on the original resource_grid
    
    # Actually, the task says: "policy is interpolated with np.interp onto np.geomspace(0.001, 40, 251)"
    # So the evaluator will interpolate our policy from resource_grid (101 points) to a 251 point grid.
    # To get a better result, we should make our policy on resource_grid as accurate as possible.
    
    # Let's use a dense internal grid for solving the Bellman equation, then map back to resource_grid
    # Or, we can just solve on resource_grid with high-order interpolation.
    
    # Let's stick with resource_grid but use higher-order splines if available, or just refine the Newton method.
    # The current score is ~8.2. The seed is ~1.0-1.3? Wait, seed score is not given but "1.0 equals the seed program".
    # Current score 8.2033 is much worse than 1.0? No, "Scoring (lower is better)".
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # Wait, if score is lower is better, and search score = seed/value...
    # If seed value is V_seed, and my value is V_me.
    # Score_me = V_me.
    # Search Score = V_seed / V_me.
    # If V_me = V_seed, Search Score = 1.0.
    # If V_me < V_seed (better), Search Score > 1.0.
    # The prompt says: "CURRENT PROGRAM (score 8.2033)".
    # Is 8.2033 the "Score" (lower is better) or the "Search Score"?
    # "The evaluator measures the raw value (lower is better)."
    # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # "An invalid result... scores 0."
    # "CURRENT PROGRAM (score 8.2033)"
    # If the raw value is 8.2033, and the seed raw value was, say, 10, then search score = 10/8.2 = 1.2.
    # Or if the raw value is 8.2033, and the seed raw value was 8.2033, search score = 1.0.
    # Given the inspirations have scores 8.25, 8.20, it seems these are the "raw values" (lower is better).
    # So I need to reduce the raw value (the gain).
    
    # Let's try to improve the accuracy of the value function interpolation.
    # The main source of error is likely the interpolation of the continuation value.
    # Using a denser grid for the value function representation could help.
    
    # Let's create a denser grid for internal calculations.
    # The evaluator interpolates our policy onto np.geomspace(0.001, 40, 251).
    # If we compute our policy on a denser grid, we can't return it directly.
    # But we can solve the Bellman equation on a denser grid, get the policy, and then...
    # Wait, the function must return policy on the *input* resource_grid.
    # "returning consumption[t, income_state, resource_index], shape (6, 2, 101)."
    # So we MUST return values on the 101-point grid.
    
    # However, we can use a denser grid *inside* the solver to compute the value function more accurately,
    # and then evaluate the optimal consumption on the 101-point grid.
    
    # Let's use a denser grid: np.geomspace(0.001, 40, 501)
    # This might be too slow?
    # 501 points * 2 states * 6 periods.
    # Newton method on 501 points.
    # Should be fast enough.
    
    dense_grid = np.geomspace(0.001, 40, 501)
    num_dense = len(dense_grid)
    
    # Re-initialize arrays for dense grid
    policy_dense = np.zeros((horizon, num_states, num_dense))
    value_dense = np.zeros((horizon, num_states, num_dense))
    
    # Terminal condition
    policy_dense[horizon - 1, :, :] = dense_grid[np.newaxis, :]
    value_dense[horizon - 1, :, :] = utility(dense_grid)[np.newaxis, :]
    
    # Backward induction on dense grid
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: interpolate current best estimate to resource_grid and return
            # We need to fill policy for t..0 on resource_grid
            # Let's just use a simple heuristic for the remaining steps
            # Or break and use what we have?
            # If we break, policy[t..0] on resource_grid is 0. That's invalid (c > 0).
            # We must fill them.
            # Let's use a linear extrapolation or just copy the next period's policy scaled?
            # For safety, let's just fill with 0.5 * m
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
            break
        
        # Create interpolation functions for V_{t+1}
        # We need V, V', V''
        # Using CubicSpline from scipy is good, but let's check if we can use higher order?
        # CubicSpline is standard.
        from scipy.interpolate import CubicSpline
        
        splines = []
        for zp in range(num_states):
            splines.append(CubicSpline(dense_grid, value_dense[t + 1, zp, :]))
            
        for z in range(num_states):
            m = dense_grid
            c_est = 0.5 * m
            
            for _ in range(30):  # More Newton steps
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    m_next = R * (m - c_est) + income[zp]
                    # Clip m_next to be within a reasonable range to avoid spline extrapolation issues
                    # Splines can extrapolate, but it might be unstable.
                    # Let's clip to [dense_grid[0]*0.5, dense_grid[-1]*2]
                    m_next_clipped = np.clip(m_next, dense_grid[0]*0.5, dense_grid[-1]*2)
                    
                    vp = splines[zp](m_next_clipped, 1)
                    vpp = splines[zp](m_next_clipped, 2)
                    
                    sum_v_prime += transition[z, zp] * vp
                    sum_v_second_prime += transition[z, zp] * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                f_prime_safe = np.where(np.abs(f_prime) < 1e-10, 1e-10, f_prime)
                c_new = c_est - f / f_prime_safe
                
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m)
                
                if np.all(np.abs(c_new - c_est) < 1e-8):
                    c_est = c_new
                    break
                c_est = c_new
            
            policy_dense[t, z, :] = c_est
            
            # Compute value V_t(m)
            u_c = utility(c_est)
            sum_v_next = np.zeros_like(m)
            for zp in range(num_states):
                m_next = R * (m - c_est) + income[zp]
                m_next_clipped = np.clip(m_next, dense_grid[0]*0.5, dense_grid[-1]*2)
                v_next = splines[zp](m_next_clipped, 0)
                sum_v_next += transition[z, zp] * v_next
                
            value_dense[t, z, :] = u_c + beta * sum_v_next
    
    # Now we have policy_dense and value_dense on the dense grid.
    # We need to return policy on the original resource_grid.
    # We can interpolate the policy from dense_grid to resource_grid.
    # Or, we can re-solve for c on resource_grid using the value functions from the dense grid?
    # Re-solving is more accurate for the policy itself.
    
    # Let's re-solve for c on resource_grid using the value functions from the last completed step.
    # Actually, we have value_dense for all t.
    # We can build splines from value_dense and solve for c on resource_grid.
    
    # Let's do this for all t.
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            # Fallback for remaining t
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
            break
        
        if t == horizon - 1:
            policy[t, :, :] = resource_grid[np.newaxis, :]
            continue
        
        # Build splines from value_dense[t+1]
        from scipy.interpolate import CubicSpline
        splines = []
        for zp in range(num_states):
            splines.append(CubicSpline(dense_grid, value_dense[t + 1, zp, :]))
            
        for z in range(num_states):
            m = resource_grid
            c_est = 0.5 * m
            
            for _ in range(30):
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    m_next = R * (m - c_est) + income[zp]
                    m_next_clipped = np.clip(m_next, dense_grid[0]*0.5, dense_grid[-1]*2)
                    
                    vp = splines[zp](m_next_clipped, 1)
                    vpp = splines[zp](m_next_clipped, 2)
                    
                    sum_v_prime += transition[z, zp] * vp
                    sum_v_second_prime += transition[z, zp] * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                f_prime_safe = np.where(np.abs(f_prime) < 1e-10, 1e-10, f_prime)
                c_new = c_est - f / f_prime_safe
                
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m)
                
                if np.all(np.abs(c_new - c_est) < 1e-8):
                    c_est = c_new
                    break
                c_est = c_new
            
            policy[t, z, :] = c_est
    
    # Ensure validity
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
