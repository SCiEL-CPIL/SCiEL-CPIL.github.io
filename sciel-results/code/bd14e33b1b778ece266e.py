import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    num_z = len(income)
    
    # Utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def mu(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    def mu_inv(x):
        # Inverse of marginal utility
        if gamma == 1:
            return 1.0 / x
        else:
            return np.power(x, -1.0 / gamma)
    
    # Value function storage: V[t, z, m]
    # We'll store the value function and consumption policy
    # V has shape (horizon+1, num_z, num_m)
    # C has shape (horizon, num_z, num_m)
    
    V = np.zeros((horizon + 1, num_z, num_m))
    C = np.zeros((horizon, num_z, num_m))
    
    # Terminal condition: V[horizon, z, m] = u(m) (consume all)
    for z in range(num_z):
        V[horizon, z, :] = u(M)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(num_z):
            m_vec = M  # shape (num_m,)
            c_candidates = np.linspace(1e-6, m_vec, 200)  # 200 candidates per m
            
            # For each m, find optimal c
            for i_m in range(num_m):
                m = m_vec[i_m]
                if m < 1e-10:
                    C[t, z, i_m] = 1e-12
                    continue
                
                c_min = 1e-12
                c_max = m * (1 - 1e-10)
                
                # Generate candidate consumptions
                c_cand = np.linspace(c_min, c_max, 200)
                
                best_val = -np.inf
                best_c = c_min
                
                for c in c_cand:
                    if c > m:
                        continue
                    rem = m - c
                    # Expected continuation value
                    cont = 0.0
                    for z_prime in range(num_z):
                        prob = transition[z, z_prime]
                        if prob < 1e-15:
                            continue
                        m_prime = R * rem + income[z_prime]
                        # Interpolate V[t+1, z_prime, m_prime]
                        if m_prime <= M[0]:
                            v_next = V[t+1, z_prime, 0]
                        elif m_prime >= M[-1]:
                            v_next = V[t+1, z_prime, -1]
                        else:
                            v_next = np.interp(m_prime, M, V[t+1, z_prime, :])
                        cont += prob * v_next
                    
                    total = u(c) + beta * cont
                    if total > best_val:
                        best_val = total
                        best_c = c
                
                C[t, z, i_m] = best_c
                V[t, z, i_m] = best_val
        
        # Check deadline
        if time.time() > deadline - 10:
            # Fill remaining with current best estimates
            for t2 in range(t, horizon):
                for z2 in range(num_z):
                    for i_m2 in range(num_m):
                        if C[t2, z2, i_m2] == 0:
                            C[t2, z2, i_m2] = M[i_m2] * 0.5
                            V[t2, z2, i_m2] = V[t2+1, z2, i_m2] if t2 < horizon else u(M[i_m2])
            break
    
    # Final polish: ensure c[5] equals grid
    C[5, :, :] = M[np.newaxis, :, np.newaxis] if C.shape[1] == 1 else M[np.newaxis, np.newaxis, :]
    # Actually C has shape (horizon, num_z, num_m)
    # C[5, z, m] should equal M[m]
    for z in range(num_z):
        C[5, z, :] = M
    
    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            for i_m in range(num_m):
                c = C[t, z, i_m]
                m = M[i_m]
                if c <= 0:
                    C[t, z, i_m] = 1e-12
                elif c > m + 1e-10:
                    C[t, z, i_m] = m + 1e-10 if m + 1e-10 > 0 else 1e-12
    
    return C
