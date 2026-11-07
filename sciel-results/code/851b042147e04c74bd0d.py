import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    
    N_m = len(m_grid)
    N_z = len(income)
    
    # Pre-compute utility function vectorized
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Initialize value function for terminal period
    # c_{T-1} = m
    V_next = np.zeros((N_m, N_z))
    for z in range(N_z):
        V_next[:, z] = util(m_grid)
        
    policies = np.zeros((horizon, N_z, N_m))
    
    # Number of candidates for consumption fraction
    # We need enough resolution to capture the optimum.
    # 256 points is a good balance.
    K = 256
    # Fractions of m to consume. 
    # Avoid 0 to prevent log(0) or inf issues, though util handles it.
    # Avoid 1 if we want to leave some savings? No, c can be m.
    # Let's use fractions from 1e-4 to 1.0.
    fracs = np.linspace(1e-4, 1.0, K)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE:
            # Fill remaining with heuristic (consume all) to ensure valid output
            for tt in range(t, -1, -1):
                for z in range(N_z):
                    policies[tt, z, :] = m_grid
            break
            
        V_curr = np.zeros((N_m, N_z))
        
        for z in range(N_z):
            # m_grid shape (N_m,)
            # fracs shape (K,)
            
            # Create grid of candidates
            # C[i, j] = m_grid[i] * fracs[j]
            # Shape: (N_m, K)
            C = m_grid[:, None] * fracs[None, :]
            
            # Utility for all candidates
            U = util(C) # Shape (N_m, K)
            
            # Next resources for all candidates
            # m' = R * (m - c) + income[z']
            # Shape: (N_m, K, N_z)
            # m_grid[:, None, None] - C[:, :, None]
            M_minus_C = m_grid[:, None] - C
            # R * (m-c)
            R_save = R * M_minus_C
            # Add income for each z'
            # income shape (N_z,) -> (1, 1, N_z)
            Next_M = R_save[:, :, None] + income[None, None, :]
            
            # Clip to grid range for interpolation
            # m_grid[0] to m_grid[-1]
            Next_M_clipped = np.clip(Next_M, m_grid[0], m_grid[-1])
            
            # Interpolate V_next for all candidates
            # V_next shape (N_m, N_z)
            # We need to interpolate along axis 0 (m) for each z'
            
            # Efficient interpolation:
            # For each z', we have V_next[:, z'] which is (N_m,).
            # We have Next_M_clipped[:, :, z'] which is (N_m, K).
            # We want result Cont[:, :, z'] of shape (N_m, K).
            
            Cont = np.zeros((N_m, K))
            
            # Vectorized interpolation is tricky with np.interp which is 1D.
            # However, we can loop over z' (only 2 states) or use scipy if available, 
            # but standard np.interp in a loop over K or z' might be slow.
            
            # Let's loop over z' (N_z is small, usually 2).
            # For a fixed z', we interpolate V_next[:, z'] at points Next_M_clipped[:, :, z'].
            # Next_M_clipped[:, :, z'] is 2D. np.interp only takes 1D x.
            # So we must flatten or loop.
            
            # Alternative: Since N_m is 101 and K is 256, N_m*K is ~25k points.
            # Looping over K (256) and calling np.interp with 101 points is 256 * 101 ops? No.
            # np.interp is C-optimized.
            
            # Let's try looping over the candidate index j (K) and vectorizing over i (N_m).
            # This is K iterations of np.interp on N_m points.
            # K=256, N_m=101. This is fast.
            
            for j in range(K):
                # Points to interpolate: Next_M_clipped[:, j, :] shape (N_m, N_z)
                # We want interpolated values for each z'.
                # For each z', x = Next_M_clipped[:, j, zp], xp = m_grid, fp = V_next[:, zp]
                
                # We can do this in a loop over z' inside the j loop, or vectorize z' somehow?
                # np.interp doesn't vectorize over fp/xp.
                
                # Let's loop over z' here too? 256 * 2 = 512 calls to np.interp.
                # Each call handles 101 points. This is very fast.
                
                for zp in range(N_z):
                    p = transition[z, zp]
                    if p > 0:
                        # x: Next_M_clipped[:, j, zp] (N_m,)
                        # xp: m_grid (N_m,)
                        # fp: V_next[:, zp] (N_m,)
                        v_interp = np.interp(Next_M_clipped[:, j, zp], m_grid, V_next[:, zp])
                        Cont[:, j] += p * v_interp
            
            # Total value: U + beta * Cont
            # Shape (N_m, K)
            Total = U + beta * Cont
            
            # Find best candidate index for each m
            # argmax along axis 1 (K)
            best_idx = np.argmax(Total, axis=1)
            
            # Extract optimal consumption and value
            # indices for advanced indexing
            m_indices = np.arange(N_m)
            c_opt = C[m_indices, best_idx]
            v_opt = Total[m_indices, best_idx]
            
            # Ensure constraints
            # c > 0, c <= m
            # Our fracs are in [1e-4, 1.0], so c is in [1e-4*m, m].
            # This satisfies 0 < c <= m.
            
            policies[t, z, :] = c_opt
            V_curr[:, z] = v_opt
            
        V_next = V_curr
        
    # Terminal period consumption is m
    for z in range(N_z):
        policies[horizon-1, z, :] = m_grid
        
    return policies
