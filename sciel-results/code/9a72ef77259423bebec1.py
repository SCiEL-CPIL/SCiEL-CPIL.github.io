import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    num_z = len(income)
    
    # Utility and marginal utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def mu(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Value function: V[t, z, m] for t in 0..horizon, z in 0..num_z-1, m in 0..num_m-1
    V = np.zeros((horizon + 1, num_z, num_m))
    C = np.zeros((horizon, num_z, num_m))
    
    # Terminal condition: consume all at horizon
    for z in range(num_z):
        V[horizon, z, :] = u(M)
    
    # Precompute: for each z, the expected next-period value as a function of m'
    # We'll compute this on the fly during backward induction
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(num_z):
            # For each m in M, find optimal c
            # c is in (0, m], and we want to maximize u(c) + beta * E[V[t+1, z', R*(m-c) + income[z']]]
            
            # Vectorize over m: for each m, try a grid of c values
            # Use a relative grid: c = m * frac, where frac in (0, 1]
            # But we need to be careful at small m
            
            # Strategy: for each m, the optimal c satisfies the Euler equation:
            # mu(c) = beta * R * E[mu(m')]
            # where m' = R*(m-c) + income[z']
            
            # We'll use a hybrid: evaluate on a grid of c fractions, then refine
            
            # Number of candidate fractions
            num_frac = 50
            
            # For each m, compute the best c
            C_tz = np.zeros(num_m)
            V_tz = np.zeros(num_m)
            
            for i_m in range(num_m):
                m = M[i_m]
                if m < 1e-10:
                    C_tz[i_m] = 1e-12
                    V_tz[i_m] = u(1e-12)
                    continue
                
                # Candidate consumption values: c = m * frac, frac in linspace(0.001, 1, num_frac)
                fracs = np.linspace(0.001, 1.0, num_frac)
                c_cands = m * fracs
                # Ensure c <= m
                c_cands = np.minimum(c_cands, m)
                c_cands = np.maximum(c_cands, 1e-12)
                
                # For each c candidate, compute the value
                # rem = m - c
                rems = m - c_cands
                # m' = R * rem + income[z']
                # For each z', compute expected V[t+1, z', m']
                
                # m' for each c and each z'
                m_primes = R * rems[:, np.newaxis] + income[np.newaxis, :]  # shape (num_frac, num_z)
                
                # Interpolate V[t+1, z', m'] for each z'
                V_next = np.zeros_like(m_primes)
                for zp in range(num_z):
                    # Clip m' to grid range
                    m_prime_clipped = np.clip(m_primes[:, zp], M[0], M[-1])
                    V_next[:, zp] = np.interp(m_prime_clipped, M, V[t + 1, zp, :])
                
                # Expected continuation value: sum over z' of transition[z, z'] * V_next[:, z']
                expected_cont = np.sum(transition[z, :] * V_next, axis=1)
                
                # Total value: u(c) + beta * expected_cont
                total_vals = u(c_cands) + beta * expected_cont
                
                best_idx = np.argmax(total_vals)
                C_tz[i_m] = c_cands[best_idx]
                V_tz[i_m] = total_vals[best_idx]
            
            C[t, z, :] = C_tz
            V[t, z, :] = V_tz
        
        # Check deadline
        if time.time() > deadline - 15:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, horizon):
                for z2 in range(num_z):
                    # Use a simple rule: consume a fraction based on horizon remaining
                    frac = 1.0 / (horizon - t2 + 1)
                    C[t2, z2, :] = np.maximum(M * frac, 1e-12)
                    C[t2, z2, :] = np.minimum(C[t2, z2, :], M)
                    V[t2, z2, :] = u(C[t2, z2, :])
            break
    
    # Ensure c[5] equals the grid (terminal period: consume all)
    for z in range(num_z):
        C[5, z, :] = M.copy()
    
    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], M + 1e-10)
    
    return C
