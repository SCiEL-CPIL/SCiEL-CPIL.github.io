import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin below 150s
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Scoring grid (as specified in task)
    # "interpolated with np.interp onto np.geomspace(0.001, 40, 251)"
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    
    # Output grid
    n_out = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Initialize Value Function for terminal period (t = horizon)
    # V_next has shape (n_fine, 2)
    V_next = np.zeros((n_fine, 2))
    
    # Policy arrays for fine grid
    # We will store optimal consumption for each t, z, m
    # Policy shape: (horizon, 2, n_fine)
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Candidate ratios
    # Use more candidates for better accuracy since vectorization is fast
    n_candidates = 1000
    ratios = np.linspace(0.001, 1.0, n_candidates)
    
    # Backward Induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5: # 5s buffer
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            # Pre-compute transition probs for z
            probs = transition[z] # [p0, p1]
            
            # Construct candidate consumption matrix: (n_fine, n_candidates)
            # C[i, k] = m_fine[i] * ratios[k]
            C = m_fine[:, np.newaxis] * ratios[np.newaxis, :]
            
            # Calculate m' for each candidate and each next state z'
            # m' = R * (m - c) + income[z']
            # m_fine[:, np.newaxis] has shape (n_fine, 1)
            # C has shape (n_fine, n_candidates)
            # m_next_0 has shape (n_fine, n_candidates)
            m_next_0 = R * (m_fine[:, np.newaxis] - C) + income[0]
            m_next_1 = R * (m_fine[:, np.newaxis] - C) + income[1]
            
            # Interpolate V_next
            # Flatten m_next arrays to use np.interp efficiently
            m_next_0_flat = m_next_0.flatten()
            m_next_1_flat = m_next_1.flatten()
            
            # np.interp handles out-of-bounds by returning boundary values.
            # m_next_min = 0 + 0.5 = 0.5. m_fine starts at 0.001. So m_next >= 0.5 is always > m_fine[0].
            # m_next_max = 1.02 * (40 - 0) + 1.5 = 42.3. m_fine ends at 40.
            # So for high m, we will extrapolate (clamp to last value).
            
            V_cont_0_flat = np.interp(m_next_0_flat, m_fine, V_next[:, 0])
            V_cont_1_flat = np.interp(m_next_1_flat, m_fine, V_next[:, 1])
            
            # Reshape back
            V_cont_0 = V_cont_0_flat.reshape(n_fine, n_candidates)
            V_cont_1 = V_cont_1_flat.reshape(n_fine, n_candidates)
            
            # Expected continuation value
            E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
            
            # Immediate utility
            U_c = utility(C)
            
            # Total Value
            Total_Value = U_c + beta * E_V_cont
            
            # Find best candidate index for each m
            idx_max = np.argmax(Total_Value, axis=1)
            
            # Get optimal consumption and value
            # Use advanced indexing
            rows = np.arange(n_fine)
            V_curr[:, z] = Total_Value[rows, idx_max]
            policy_fine[t, z, :] = C[rows, idx_max]
            
        V_next = V_curr
        
    # Interpolate policy from fine grid (251) to output grid (101)
    # We need to return shape (6, 2, 101)
    # The evaluator interpolates this result onto the 251 grid.
    # So we want our 101-point values to be a good approximation of the 251-point optimal policy.
    
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            # policy_fine[t, z, :] is the optimal consumption on m_fine
            # resource_grid is the target grid
            consumption_out[t, z, :] = np.interp(resource_grid, m_fine, policy_fine[t, z, :])
            
    return consumption_out
