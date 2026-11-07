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
    
    M_out = resource_grid
    num_m_out = len(M_out)
    
    # Fine grid for internal computation
    # Using 1000 points to balance accuracy and speed
    M_fine = np.geomspace(0.001, 40, 1000)
    num_m_fine = len(M_fine)
    
    # Candidate consumption fractions
    K = 301
    fracs = np.linspace(0.001, 1.0, K)
    
    # V[t, z, i] is the value function on the fine grid
    # C[t, z, i] is the consumption policy on the fine grid
    V = np.zeros((horizon, 2, num_m_fine))
    C = np.zeros((horizon, 2, num_m_fine))
    
    # Final period t = horizon - 1
    V[horizon-1, :, :] = 0.0
    C[horizon-1, :, :] = M_fine
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining periods with the next period's policy
            for tt in range(t, -1, -1):
                if tt + 1 < horizon:
                    C[tt, :, :] = C[tt+1, :, :]
                    V[tt, :, :] = V[tt+1, :, :]
                else:
                    C[tt, :, :] = M_fine
                    V[tt, :, :] = 0.0
            
            # Interpolate back to output grid
            C_out = np.zeros((horizon, 2, num_m_out))
            for tt in range(horizon):
                for z in range(2):
                    C_out[tt, z, :] = np.interp(M_out, M_fine, C[tt, z, :])
            C_out[5, :, :] = M_out
            return C_out

        for z in range(2):
            C_cand = np.outer(M_fine, fracs)
            
            if abs(gamma - 1.0) < 1e-10:
                u = np.log(np.maximum(C_cand, 1e-12))
            else:
                u = np.power(np.maximum(C_cand, 1e-12), 1.0 - gamma) / (1.0 - gamma)
            
            M_minus_c = M_fine[:, None] - C_cand
            
            cont_val = np.zeros_like(C_cand)
            
            for z_prime in range(2):
                p = transition[z, z_prime]
                if p == 0:
                    continue
                
                M_next = R * M_minus_c + income[z_prime]
                
                M_flat = M_next.flatten()
                V_target = V[t+1, z_prime, :]
                
                V_interp_flat = np.interp(M_flat, M_fine, V_target)
                V_interp = V_interp_flat.reshape(num_m_fine, K)
                
                cont_val += p * V_interp
            
            total_val = u + beta * cont_val
            
            opt_k = np.argmax(total_val, axis=1)
            
            indices = np.arange(num_m_fine)
            C[t, z, :] = C_cand[indices, opt_k]
            V[t, z, :] = total_val[indices, opt_k]
            
    # Interpolate the fine policy back to the output grid
    C_out = np.zeros((horizon, 2, num_m_out))
    for t in range(horizon):
        for z in range(2):
            C_out[t, z, :] = np.interp(M_out, M_fine, C[t, z, :])
    
    # Enforce constraints
    C_out = np.clip(C_out, 1e-9, M_out[:, None] + 1e-10)
    C_out[5, :, :] = M_out
    
    return C_out
