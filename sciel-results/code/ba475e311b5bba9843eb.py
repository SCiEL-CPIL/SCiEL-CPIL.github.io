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

    # Use a very dense grid for high accuracy in interpolation
    # The evaluator interpolates onto 251 points. Solving on 1000 points
    # allows for extremely accurate representation of the policy.
    n_dense = 1000
    dense_grid = np.geomspace(0.001, 40, n_dense)
    
    # We will store the value function V_t(m, z) on the dense grid.
    # Shape: (horizon, 2, n_dense)
    V = np.zeros((horizon, 2, n_dense))
    
    # Policy will be derived from V at the end, or we can store it.
    # Storing policy directly is easier for the final output requirement.
    # However, Value Function Iteration naturally produces V.
    # Let's compute V first, then derive policy.
    
    # Terminal condition: V_T(m, z) = u(m)
    # But the task says c[5] (t=5, last period) equals the grid.
    # So at t=5, c=m. This implies V_5(m, z) = u(m).
    # And for t < 5, we maximize.
    
    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Initialize V for the last period (t=5)
    V[5, :, :] = get_utility(dense_grid, gamma)[np.newaxis, :]
    
    # Backward induction for t = 4, 3, 2, 1, 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: if we run out of time, we need to fill the remaining V.
            # A crude approximation is to assume c=m for all remaining periods.
            for tt in range(t, -1, -1):
                V[tt, :, :] = get_utility(dense_grid, gamma)[np.newaxis, :]
            break
            
        # V_t(m, z) = max_c { u(c) + beta * E[V_{t+1}(R(m-c) + income[z'], z') | z] }
        
        # We need to solve this maximization for all m in dense_grid and z in {0, 1}.
        # To do this vectorized, we can't easily iterate over c for each m.
        # Instead, we can use a discrete grid for c as well, or use a property of the solution.
        # However, a full grid search over c for each m is O(n_dense * n_c).
        # With n_dense=1000 and n_c=1000, that's 1M operations per t,z.
        # 5 periods * 2 states * 1M = 10M operations. NumPy can handle this in <1s.
        
        # Let's define a candidate grid for c.
        # c must be in [0, m]. Since m varies, we can't use a single fixed c grid for all m easily 
        # unless we mask invalid c > m.
        # Alternatively, we can parameterize c as a fraction of m? No, policy is not strictly linear.
        
        # Let's use a dense c_grid for each m.
        # Actually, we can vectorize the maximization over a set of candidate c values.
        # Let c_candidates be a grid of size n_c.
        # For a given m, valid c are c <= m.
        
        # Strategy:
        # 1. Create a meshgrid of m (n_dense) and c (n_c).
        # 2. Compute u(c) + beta * E[V_next(m')] for all pairs.
        # 3. Mask out c > m.
        # 4. Take max over c dimension.
        
        n_c = 500 # Number of candidate consumption levels
        # We need c to range from 0 to max(m).
        # But for small m, we only need small c.
        # Using a global c_grid from 0 to 40 is wasteful for small m but simple.
        # Let's use a c_grid that matches the dense_grid range.
        c_grid = np.geomspace(1e-4, 40, n_c)
        
        # V_next is shape (2, n_dense)
        V_next = V[t+1, :, :]
        
        V_current = np.zeros((2, n_dense))
        
        for z in range(2):
            # m_vals: (n_dense,)
            # c_vals: (n_c,)
            
            # We want to compute for each m_i, the best c_j.
            # m' = R(m_i - c_j) + income[z']
            # E[V_next(m', z')] = sum_{z'} P(z'|z) * V_next(z', m')
            
            # To vectorize efficiently without a massive (2, n_dense, n_c, 2) array,
            # we can loop over z' (only 2 iterations) and accumulate.
            
            # Let's precompute the interpolation for V_next for the specific m' values.
            # m' depends on m_i and c_j.
            
            # Meshgrid:
            # m_i shape: (n_dense, 1)
            # c_j shape: (1, n_c)
            m_mesh = dense_grid[:, np.newaxis] # (n_dense, 1)
            c_mesh = c_grid[np.newaxis, :]    # (1, n_c)
            
            # Current utility: u(c_j)
            u_c = get_utility(c_mesh, gamma) # (1, n_c)
            
            # Continuation value
            # We need E[V_next] for each (m_i, c_j)
            # E[V_next] = P0 * V_next(0, R(m_i - c_j) + inc0) + P1 * V_next(1, R(m_i - c_j) + inc1)
            
            # Compute m' for z'=0
            m_next_0 = R * (m_mesh - c_mesh) + income[0] # (n_dense, n_c)
            # Compute m' for z'=1
            m_next_1 = R * (m_mesh - c_mesh) + income[1] # (n_dense, n_c)
            
            # Interpolate V_next(0, m_next_0) and V_next(1, m_next_1)
            # np.interp requires 1D arrays. We can use np.apply_along_axis or reshape.
            # Or, since dense_grid is sorted, we can use searchsorted for speed if we just need values.
            # But np.interp is vectorized along the last axis if we reshape? No.
            # Let's use a helper that flattens, interpolates, and reshapes.
            # Or simply, since n_dense=1000 and n_c=500, the array is 500k elements.
            # np.interp on 500k elements is fast.
            
            # Flatten m_next_0
            m_next_0_flat = m_next_0.flatten()
            V_next_0_flat = np.interp(m_next_0_flat, dense_grid, V_next[0, :])
            V_next_0 = V_next_0_flat.reshape(n_dense, n_c)
            
            m_next_1_flat = m_next_1.flatten()
            V_next_1_flat = np.interp(m_next_1_flat, dense_grid, V_next[1, :])
            V_next_1 = V_next_1_flat.reshape(n_dense, n_c)
            
            # Expected continuation value
            exp_V = transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1
            
            # Total value
            total_V = u_c + beta * exp_V # (n_dense, n_c) via broadcasting
            
            # Mask invalid c > m
            # c_mesh is (1, n_c), m_mesh is (n_dense, 1)
            # c > m means c_mesh > m_mesh
            mask = c_mesh > m_mesh
            total_V[mask] = -np.inf
            
            # Find max over c (axis 1)
            # We also need the argmax to get the policy later, but for now just V.
            V_current[z, :] = np.max(total_V, axis=1)
            
            # We could store the policy here to avoid re-computing it later,
            # but re-computing it at the end is also fine and keeps memory lower 
            # if we don't store policy for all t. 
            # Actually, storing policy is (6, 2, 1000) which is small.
            # Let's store the argmax index to derive policy later.
            # But we need the policy on the final 101 grid.
            # It's better to compute the policy on the dense grid now and interpolate.
            
            # argmax_c gives index in c_grid
            idx_max = np.argmax(total_V, axis=1)
            policy_dense = c_grid[idx_max]
            
            # Store policy for this t, z on dense grid
            # We need an array to hold this.
            if not hasattr(V, 'policy'):
                V.policy = np.zeros((horizon, 2, n_dense))
            V.policy[t, z, :] = policy_dense

    # Now we have V.policy on the dense grid (1000 points).
    # We need to return consumption on resource_grid (101 points).
    
    consumption = np.zeros((horizon, 2, n_m))
    
    for t in range(horizon):
        for z in range(2):
            consumption[t, z, :] = np.interp(resource_grid, dense_grid, V.policy[t, z, :])
            
    # Final checks
    # 1. Finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=40, neginf=1e-12)
    
    # 2. 0 < c <= m + 1e-10
    # Clip lower bound
    consumption = np.maximum(consumption, 1e-12)
    # Clip upper bound
    max_c = resource_grid[np.newaxis, np.newaxis, :] + 1e-10
    consumption = np.minimum(consumption, max_c)
    
    # 3. c[5] equal to the grid within 1e-9
    consumption[5, :, :] = resource_grid[np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
