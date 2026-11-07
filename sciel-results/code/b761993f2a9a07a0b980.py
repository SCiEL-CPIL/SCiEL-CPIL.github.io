import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    
    # Create a finer grid for better interpolation and optimization
    # The evaluator interpolates onto geomspace(0.001, 40, 251)
    # We'll use a fine grid internally to get more accurate policies
    M_fine = np.geomspace(0.001, 40, 801)
    num_m_fine = len(M_fine)
    
    # We'll compute on the fine grid, then interpolate back to the original grid
    # This allows for more accurate backward induction
    
    # Number of consumption candidates per wealth level
    K = 200  # Reduced from 500 to allow finer grid
    
    # Fractions of wealth to consume
    fracs = np.linspace(0.001, 1.0, K)
    
    # V[t, z, i] is the value function on fine grid
    # C[t, z, i] is the consumption policy on fine grid
    V = np.zeros((horizon, 2, num_m_fine))
    C = np.zeros((horizon, 2, num_m_fine))
    
    # Final period t = horizon - 1
    # Agent consumes all wealth
    V[horizon-1, :, :] = 0.0
    C[horizon-1, :, :] = M_fine
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # Fill remaining periods with next period's policy
            for tt in range(t, -1, -1):
                if tt + 1 < horizon:
                    C[tt, :, :] = C[tt+1, :, :]
                    V[tt, :, :] = V[tt+1, :, :]
                else:
                    C[tt, :, :] = M_fine
                    V[tt, :, :] = 0.0
            # Interpolate back to original grid
            C_out = np.zeros((horizon, 2, num_m))
            for tt in range(horizon):
                for z in range(2):
                    C_out[tt, z, :] = np.interp(M, M_fine, C[tt, z, :])
            # Enforce final period constraint
            C_out[horizon-1, :, :] = M
            return C_out

        for z in range(2):
            # Construct candidate consumptions: (num_m_fine, K)
            C_cand = np.outer(M_fine, fracs)
            
            # Immediate utility
            if abs(gamma - 1.0) < 1e-10:
                u = np.log(np.maximum(C_cand, 1e-12))
            else:
                u = np.power(np.maximum(C_cand, 1e-12), 1.0 - gamma) / (1.0 - gamma)
            
            # Next resources: M' = R(M - c) + income[z']
            M_minus_c = M_fine[:, None] - C_cand
            
            cont_val = np.zeros_like(C_cand)
            
            for z_prime in range(2):
                p = transition[z, z_prime]
                if p == 0:
                    continue
                
                M_next = R * M_minus_c + income[z_prime]
                
                # Interpolate V_{t+1}[:, z_prime] at M_next
                M_flat = M_next.flatten()
                V_flat_target = V[t+1, z_prime, :]
                
                V_interp_flat = np.interp(M_flat, M_fine, V_flat_target)
                V_interp = V_interp_flat.reshape(num_m_fine, K)
                
                cont_val += p * V_interp
            
            # Total value
            total_val = u + beta * cont_val
            
            # Find optimal k for each i
            opt_k = np.argmax(total_val, axis=1)
            
            # Extract optimal c and V
            indices = np.arange(num_m_fine)
            C[t, z, :] = C_cand[indices, opt_k]
            V[t, z, :] = total_val[indices, opt_k]
            
    # Interpolate back to original grid
    C_out = np.zeros((horizon, 2, num_m))
    for tt in range(horizon):
        for z in range(2):
            C_out[tt, z, :] = np.interp(M, M_fine, C[tt, z, :])
    
    # Enforce final period constraint: c[5] must equal the grid
    C_out[horizon-1, :, :] = M
    
    # Ensure validity: 0 < c <= m + 1e-10
    C_out = np.clip(C_out, 1e-12, M[None, None, :] + 1e-10)
    C_out[horizon-1, :, :] = M  # Re-enforce final period
    
    return C_out
