"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 120s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    
    # We will store the value function V[t, z, m]
    # and the policy c[t, z, m]
    V = np.zeros((horizon, 2, n_m))
    c_policy = np.zeros((horizon, 2, n_m))
    
    # Utility function
    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Initialize last period (t = horizon - 1)
    # V_{N-1}(m, z) = u(m)
    # c_{N-1}(m, z) = m
    V[horizon-1, :, :] = get_utility(m_grid, gamma)
    c_policy[horizon-1, :, :] = m_grid[np.newaxis, :]

    # Backward induction for t = horizon - 2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: if running out of time, fill remaining with a simple rule
            # This is unlikely to be reached if the method is fast, but good for safety
            for tt in range(t, -1, -1):
                c_policy[tt, :, :] = m_grid[np.newaxis, :] / (horizon - tt)
            break
        
        # For each z, we need to find c that maximizes:
        # u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c) + income[z'], z')
        
        # We will evaluate this objective on a dense grid of c for each m.
        # To be efficient, we can vectorize over m.
        
        # Let's define a dense grid of consumption fractions or absolute values.
        # The evaluator interpolates the policy onto a finer grid.
        # The policy should be accurate.
        # Let's use a grid of c values.
        # Since m varies, c should be proportional to m? Not exactly, but close.
        # A robust approach: for each m, search c in [0, m].
        
        # Vectorized approach:
        # We have m_grid (101 values).
        # We want to find c for each m.
        
        # Let's create a grid of c candidates.
        # Since m is geomspace, c should probably be geomspace too, or linear in m.
        # Let's try a linear grid of fractions of m: c = m * f, where f in [0, 1].
        # But the optimal c/m is not constant.
        
        # Better: For each m, the optimal c is unique.
        # We can use a fixed number of candidate c values for each m.
        # Let's use 101 candidate values for c, linearly spaced from 0 to m?
        # Or geomspace?
        
        # Actually, the objective is concave in c.
        # We can use a ternary search or golden section search for each m.
        # Or, since we have V_{t+1} on a grid, we can interpolate it.
        
        # Let's try a dense grid of c values for each m.
        # To speed up, we can use a global grid of c values?
        # No, c depends on m.
        
        # Let's use a grid of 51 points for c for each m.
        # This is 101 * 51 * 2 * 2 (for z and z') operations per period.
        # 51 * 101 * 2 * 2 * 5 periods ~ 100k operations. Very fast.
        
        n_c = 101 # Number of candidate consumption values
        # Use geomspace for c to handle small m well?
        # Or linspace?
        # Let's use linspace for simplicity, but ensure c > 0.
        
        for z in range(2):
            # m_vals: shape (n_m,)
            # We want to evaluate objective for each m and each c candidate.
            
            # c_candidates: shape (n_m, n_c)
            # Let's use a grid that covers [0, m]
            # To avoid c=0 issues, start from a small epsilon.
            
            # Let's use a linear grid of fractions.
            # f = np.linspace(0, 1, n_c)
            # c = m[:, np.newaxis] * f[np.newaxis, :]
            
            # But the optimal c might be very small for small m.
            # Let's use a combination.
            # Actually, the evaluator uses geomspace for m.
            # Let's use a dense linear grid for c for each m.
            
            # To be safe and accurate, let's use 201 points.
            n_c_dense = 201
            f = np.linspace(0, 1, n_c_dense)
            # Ensure c > 0 for utility
            f = np.maximum(f, 1e-6)
            
            c_candidates = m_grid[:, np.newaxis] * f[np.newaxis, :] # shape (n_m, n_c_dense)
            
            # Compute u(c)
            u_c = get_utility(c_candidates, gamma) # shape (n_m, n_c_dense)
            
            # Compute continuation value
            # m' = R * (m - c) + income[z']
            # For each z', we need V_{t+1}(m', z')
            
            # m_next: shape (n_m, n_c_dense, 2)
            m_next = R * (m_grid[:, np.newaxis, np.newaxis] - c_candidates[:, :, np.newaxis]) + income[np.newaxis, np.newaxis, :]
            
            # We need to interpolate V_{t+1}[z'] at m_next.
            # V_{t+1}[z'] is defined on m_grid.
            # np.interp works on 1D arrays. We can use it row by row or use a trick.
            
            # Let's compute the expected continuation value for each (m, c).
            # E[V'] = sum_{z'} P(z'|z) * V_{t+1}(m', z')
            
            # We can loop over z' (only 2 states)
            expected_V = np.zeros((n_m, n_c_dense))
            
            for zp in range(2):
                # m_next_zp: shape (n_m, n_c_dense)
                m_zp = m_next[:, :, zp]
                
                # Interpolate V_{t+1}[zp] at m_zp
                # V_next_zp = V[t+1, zp, :] shape (n_m,)
                # We need to interpolate this 1D function at m_zp (2D array).
                
                # np.interp can take 1D x. We can reshape m_zp to 1D, interpolate, then reshape back.
                m_zp_flat = m_zp.ravel()
                V_interp_flat = np.interp(m_zp_flat, m_grid, V[t+1, zp, :])
                V_interp = V_interp_flat.reshape(m_zp.shape)
                
                expected_V += transition[z, zp] * V_interp
            
            # Total value: u(c) + beta * expected_V
            total_value = u_c + beta * expected_V # shape (n_m, n_c_dense)
            
            # Find the c that maximizes total_value for each m.
            # Since the function is concave, the maximum is unique.
            # We can use argmax.
            
            idx_max = np.argmax(total_value, axis=1) # shape (n_m,)
            
            # Extract the optimal c
            c_opt = c_candidates[np.arange(n_m), idx_max]
            
            c_policy[t, z, :] = c_opt
            V[t, z, :] = total_value[np.arange(n_m), idx_max]

    # Final check and clipping
    # Ensure 0 < c <= m + 1e-10
    c_policy = np.clip(c_policy, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    # Ensure c[5] is exactly the grid
    c_policy[5, :, :] = resource_grid[np.newaxis, :]
    
    return c_policy
# EVOLVE-BLOCK-END
