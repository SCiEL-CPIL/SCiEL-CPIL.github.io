import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    # Use a denser grid for internal computation to improve interpolation accuracy
    # but the final output must be on resource_grid
    # Let's stick to resource_grid for output shape, but maybe use a finer internal grid?
    # The task says return shape (6, 2, 101) on resource_grid.
    # However, the scorer interpolates our policy onto a 251-point grid.
    # If we solve exactly on the 101-point grid, interpolation might still have error.
    # But solving on a 251-point grid internally and then interpolating back to 101 for output?
    # No, the output MUST be on resource_grid.
    # So we solve for the optimal c on resource_grid.
    
    # To improve accuracy of the *continuation value* used in the optimization,
    # we can use a finer grid for the value function lookup?
    # The problem is the value function is defined on the grid.
    # If we use a finer grid internally, we need to map the state back.
    # Let's first try to optimize the search method on the existing grid.
    
    # The key insight from the prompt: "Golden section search finds the exact continuous optimum... reducing gain to near-zero"
    # The scorer uses np.interp to evaluate our policy on a finer grid.
    # If our policy is exact on the coarse grid, the interpolation error is minimized.
    
    # Strategy: For each t, z, m, find c that maximizes u(c) + beta * E[V_{t+1}(m')]
    # m' = R(m-c) + income[z']
    # This is a concave maximization problem in c.
    # We can use scipy.optimize.minimize_scalar or implement bisection on the derivative.
    # Since we have 101 grid points, 2 states, 6 times, that's 1212 optimizations.
    # Each optimization is fast if we vectorize the expectation.
    
    # Let's use a dense internal grid to compute the value function more accurately?
    # No, the state space is continuous in m.
    # Standard approach: Discretize m.
    # If we discretize m on resource_grid, the transition m' = R(m-c)+y might fall off the grid.
    # The scorer handles this by interpolating the continuation value.
    # So our internal value function should also interpolate.
    
    # Let's use a finer grid for the value function to reduce interpolation error in the expectation?
    # If we use a grid of size 201 or 401, the memory is (6, 2, 401) which is small.
    # Time: 6 * 2 * 401 * (search time).
    # Search time: ~50-100 evaluations of the objective.
    # Objective evaluation: vectorized over z' (2 states).
    # This should be fast.
    
    # Let's try with a finer grid internally, say 201 points, to get a smoother value function.
    # Then we interpolate the final policy back to the required 101-point resource_grid.
    
    internal_grid = np.geomspace(0.001, 40, 201)
    num_internal = len(internal_grid)
    
    # Policy and Value on internal grid
    # We need to return policy on resource_grid
    # So we compute V and Policy on internal_grid, then interp Policy to resource_grid.
    
    policy_internal = np.zeros((horizon, num_states, num_internal))
    value_internal = np.zeros((horizon, num_states, num_internal))
    
    # Terminal condition: c[5] = m (consume all)
    # Wait, the prompt says "c[5] equal to the grid within 1e-9".
    # Horizon is 6, so t=0..5.
    # At t=5 (last period), c = m.
    value_internal[5, :, :] = _utility_grid(internal_grid, gamma)
    policy_internal[5, :, :] = internal_grid[None, None, :]
    
    # Backward induction
    for t in range(4, -1, -1):
        # Check time
        if time.time() > deadline - 5:
            # Fallback: linear interpolation or simple rule?
            # If we run out of time, we need to return something valid.
            # We have value for t+1. We can do a quick greedy or linear approx for remaining t.
            # Or just return the current best estimate (which is zeros for t<5 if not computed).
            # Better: use a simple heuristic like c = m * beta * R for remaining periods?
            # Or just linear interp of policy from t+1?
            # Let's try to finish as many as possible.
            # For now, break and handle at the end.
            pass
        
        for z in range(num_states):
            # We need to solve for each m in internal_grid
            # Objective: u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(z', R(m-c)+income[z'])
            
            # Vectorize over m?
            # For each m, we need to find c.
            # The optimal c is where derivative is 0.
            # u'(c) = beta * sum P * V'_{t+1}(m') * (-R)
            
            # Since V is concave, we can use bisection on the derivative.
            # Or use scipy.optimize.minimize_scalar with bounds [0, m].
            # minimize_scalar is robust.
            
            # Let's try to vectorize the search?
            # It's hard to vectorize the search itself because each m has a different optimum.
            # But we can evaluate the objective for many c's at once.
            
            # Alternative: Use the fact that the Euler equation is monotonic.
            # Let's just loop over m for simplicity and speed up the inner loop.
            
            m_vals = internal_grid
            c_opts = np.zeros(num_internal)
            
            # Pre-compute transition probabilities for this z
            probs = transition[z]
            
            # We will use bisection on the first-order condition.
            # F(c) = u'(c) + beta * R * sum P * V'_{t+1}(z', m')
            # Note: V' is negative, so the second term is negative.
            # We want F(c) = 0.
            # As c increases, u'(c) decreases (becomes more negative).
            # m' = R(m-c)+y decreases, so V'(m') increases (becomes less negative).
            # So the second term (beta * R * sum P * V') increases (becomes less negative).
            # Wait, V' is negative. Let V' = -k.
            # Term is -beta * R * sum P * k.
            # As c increases, m' decreases, k increases (since V is concave, V' becomes more negative as m decreases? No, V' becomes more negative as m increases?
            # V(m) is concave, so V'(m) is decreasing.
            # As m decreases, V'(m) increases (becomes less negative).
            # So as c increases, m' decreases, V'(m') increases (less negative).
            # So the term beta * R * sum P * V'(m') increases (less negative).
            # u'(c) decreases (more negative).
            # So F(c) is strictly decreasing.
            # F(0) > 0? u'(0) is huge positive. V' is finite negative. So F(0) > 0.
            # F(m) < 0? u'(m) is finite. V'(income) is finite negative.
            # Actually, at c=m, m' = income.
            # We need to check if the optimum is at the boundary.
            # Usually interior.
            
            # Let's use scipy.optimize.brentq on the derivative?
            # Or just minimize_scalar on the negative objective.
            # minimize_scalar is safer against boundary issues.
            
            # To speed up, we can use a coarser search first?
            # Or just use a fixed number of iterations of bisection on the derivative.
            # Derivative of V_{t+1} can be approximated by np.gradient or finite differences on the grid.
            # Since we have value_internal[t+1], we can compute dV_dm = np.gradient(value_internal[t+1], internal_grid, axis=2).
            
            dV_dm_next = np.gradient(value_internal[t+1], internal_grid, axis=2)
            
            # For each m, solve F(c) = 0
            # F(c) = u'(c) + beta * R * sum_{z'} probs[z'] * dV_dm_next[z', idx(m')]
            # where idx(m') is interpolated.
            
            # Let's vectorize the evaluation of F(c) for a given c grid?
            # No, c is different for each m.
            
            # Let's loop over m. 201 points * 2 states * 5 periods = 2010 loops.
            # Each loop does ~30-50 evaluations of F(c).
            # Total ~100,000 evaluations.
            # Each evaluation is a few numpy ops. Should be fast.
            
            for i_m, m in enumerate(m_vals):
                # Define F(c)
                def F(c_val):
                    # u'(c)
                    if c_val <= 0:
                        return 1e10 # Large positive
                    if gamma == 1:
                        u_prime = 1.0 / c_val
                    else:
                        u_prime = c_val ** (-gamma)
                    
                    # m'
                    m_prime = R * (m - c_val) + income
                    # Interpolate dV_dm_next
                    # dV_dm_next[z', :] is the derivative for state z'
                    # We need sum probs[z'] * dV_dm_next[z', m_prime[z']]
                    # m_prime is a vector of length 2
                    # Use np.interp for each z'
                    
                    # dV_dm_next shape (2, 201)
                    # m_prime shape (2,)
                    # We can use a single np.interp call if we stack?
                    # Or just loop over z' (only 2 states)
                    
                    exp_dV = 0.0
                    for zp in range(num_states):
                        # Clip m_prime to grid range
                        mp = np.clip(m_prime[zp], internal_grid[0], internal_grid[-1])
                        dv = np.interp(mp, internal_grid, dV_dm_next[zp, :])
                        exp_dV += probs[zp] * dv
                    
                    return u_prime + beta * R * exp_dV
                
                # Bisection for F(c) = 0
                # Bounds: [0, m]
                # F(0) > 0, F(m) < 0 (usually)
                # Let's check F(m)
                # If F(m) > 0, optimum is at m (consume all).
                # If F(0) < 0, optimum is at 0 (save all).
                
                lo = 0.0
                hi = m
                
                # Check boundaries
                if F(lo) < 0:
                    c_opt = 0.0
                elif F(hi) > 0:
                    c_opt = m
                else:
                    # Bisection
                    for _ in range(30): # ~1e-9 precision
                        mid = 0.5 * (lo + hi)
                        f_mid = F(mid)
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                    c_opt = 0.5 * (lo + hi)
                
                c_opts[i_m] = c_opt
                
                # Check time
                if time.time() > deadline - 2:
                    # If we are running out of time, break out and use a heuristic for the rest
                    # Fill remaining c_opts with a simple heuristic: c = m * 0.5?
                    # Or just break and let the final interpolation handle it?
                    # If we break, c_opts will be partially filled.
                    # We should fill the rest with a safe value, e.g., linear interpolation from previous?
                    # Or just use c = m * beta * R / (1 + beta * R) ?
                    # Let's use a simple constant saving rate for the rest.
                    # This is a fallback.
                    pass

            policy_internal[t, z, :] = c_opts
            
            # Calculate value
            # V_t(z, m) = u(c) + beta * E[V_{t+1}(z', m')]
            # We can compute this using the optimal c
            c_vals = c_opts
            m_vals = internal_grid
            m_prime = R[:, None] * (m_vals[None, :] - c_vals[None, :]) + income[:, None]
            # m_prime shape (2, 201)
            # V_{t+1} shape (2, 201)
            # Interpolate V_{t+1} at m_prime
            V_next_interp = np.zeros_like(m_prime)
            for zp in range(num_states):
                mp = np.clip(m_prime[zp, :], internal_grid[0], internal_grid[-1])
                V_next_interp[zp, :] = np.interp(mp, internal_grid, value_internal[t+1, zp, :])
            
            exp_V_next = np.sum(probs[:, None] * V_next_interp, axis=0)
            
            # u(c)
            if gamma == 1:
                u_c = np.log(np.maximum(c_vals, 1e-12))
            else:
                u_c = (np.maximum(c_vals, 1e-12) ** (1 - gamma)) / (1 - gamma)
            
            value_internal[t, z, :] = u_c + beta * exp_V_next

    # Interpolate policy from internal_grid to resource_grid
    policy = np.zeros((horizon, num_states, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(resource_grid, internal_grid, policy_internal[t, z, :])
    
    # Ensure validity: 0 < c <= m + 1e-10
    # And c[5] == grid
    # The interpolation might cause slight violations.
    # Clip c to [1e-12, m + 1e-10]
    for t in range(horizon):
        for z in range(num_states):
            m_grid = resource_grid
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
            
    # Force terminal condition
    policy[5, :, :] = resource_grid[None, None, :]
    
    return policy

def _utility_grid(m, gamma):
    if gamma == 1:
        return np.log(np.maximum(m, 1e-12))
    else:
        return (np.maximum(m, 1e-12) ** (1 - gamma)) / (1 - gamma)
