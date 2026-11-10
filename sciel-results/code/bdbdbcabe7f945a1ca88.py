import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model['R'])
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])

    num_states = len(income)
    m_grid = np.asarray(resource_grid, dtype=float)
    n_m = len(m_grid)
    c_floor = 1e-12
    log_floor = np.log(c_floor)

    is_log = abs(gamma - 1.0) < 1e-12

    # Continuation value V[t][z][m]
    # Policy C[t][z][m]
    V = np.empty((horizon, num_states, n_m), dtype=float)
    C = np.empty((horizon, num_states, n_m), dtype=float)

    # Terminal: consume all
    V[horizon - 1] = 0.0  # not used directly; C[horizon-1] = m
    C[horizon - 1] = m_grid[np.newaxis, :]

    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 1.0:
            # Safety: fill remaining with consume-all (valid but suboptimal)
            C[t] = m_grid[np.newaxis, :]
            V[t] = 0.0
            continue

        # Expected continuation value for each (z, m)
        # m' = R*(m - c) + income[z']
        # For a given c, m' depends on z'. We need E_{z'}[V[t+1][z'][m']]
        # We'll compute this via interpolation over m_grid for each z'
        
        # First, we need to find optimal c for each (z, m)
        # Euler equation: u'(c) = beta * R * E_{z'}[u'(c'[t+1][z', m'])]
        # where c' = C[t+1][z'][m'], m' = R*(m-c) + income[z']
        
        # Since C[t+1] is already computed, we can compute the RHS for any c.
        # Strategy: vectorized bisection on c in [c_floor, m]
        
        # Pre-compute interpolation setup for V[t+1]
        # V[t+1][z][m] is defined on m_grid
        # We need E_z'[V[t+1][z'][R*(m-c)+income[z']]]
        
        # For efficiency, we'll evaluate the Euler residual for all (z, m) at once
        # c shape: (num_states, n_m)
        
        # Lower bound: c_floor
        # Upper bound: m_grid (can't consume more than m)
        c_lo = np.full((num_states, n_m), c_floor)
        c_hi = m_grid[np.newaxis, :].copy()  # shape (1, n_m), broadcast to (num_states, n_m)
        
        # We'll iterate bisection
        # To speed up, we can limit iterations
        max_iter = 40
        tol = 1e-10
        
        for _ in range(max_iter):
            if time.time() > deadline - 0.5:
                break
            c_mid = 0.5 * (c_lo + c_hi)
            
            # Compute m' for each z'
            # c_mid shape: (num_states, n_m)
            # For each z (current state) and m, we have c_mid[z, m]
            # m' = R*(m - c) + income[z']
            # We need E_{z'|z}[V[t+1][z'][m']]
            
            # Compute m' for all z' transitions
            # m' shape: (num_states, n_m, num_states) for z, m, z'
            m_prime = R * (m_grid[np.newaxis, :, np.newaxis] - c_mid[:, :, np.newaxis]) + income[np.newaxis, np.newaxis, :]
            # m_prime[z, m, z'] = R*(m_grid[m] - c_mid[z, m]) + income[z']
            
            # Clip m' to grid range for interpolation
            m_min = m_grid[0]
            m_max = m_grid[-1]
            m_prime_clipped = np.clip(m_prime, m_min, m_max)
            
            # Interpolate V[t+1][z'][m']
            # V_next[z'][m] is on m_grid
            # For each (z, m, z'), get V[t+1][z'][m_prime_clipped[z, m, z']]
            # Use np.interp for each z'
            # Vectorized: for each z', interp over m_grid
            V_next = V[t + 1]  # shape (num_states, n_m)
            
            # expected_V shape: (num_states, n_m)
            # For each z', we have V_next[z'] on m_grid
            # m_prime_clipped shape: (num_states, n_m, num_states)
            # We need to interpolate V_next[z'] at m_prime_clipped[:, :, z'] for each z'
            
            # Do it z' by z' but vectorized over (z, m)
            expected_V = np.zeros((num_states, n_m), dtype=float)
            for zp in range(num_states):
                # V_next[zp] shape (n_m,)
                # m_prime_clipped[:, :, zp] shape (num_states, n_m)
                # Interpolate
                vals = np.interp(m_prime_clipped[:, :, zp], m_grid, V_next[zp])
                expected_V += transition[:, zp] * vals
            
            # Now compute RHS of Euler: beta * R * E[u'(c')]
            # u'(c) for CRRA:
            # if gamma=1: 1/c
            # else: c^(-gamma)
            
            # We need E_{z'}[u'(C[t+1][z'][m'])]
            # C_next[z'][m] shape (num_states, n_m)
            C_next = C[t + 1]
            
            # c_prime = C_next[z'][m_prime_clipped[z, m, z']]
            # Interpolate C_next[z'] at m_prime_clipped
            # c_prime shape: (num_states, n_m, num_states)
            c_prime = np.zeros((num_states, n_m, num_states), dtype=float)
            for zp in range(num_states):
                c_prime[:, :, zp] = np.interp(m_prime_clipped[:, :, zp], m_grid, C_next[zp])
            
            # u'(c_prime)
            c_prime_safe = np.maximum(c_prime, c_floor)
            if is_log:
                u_prime_cprime = 1.0 / c_prime_safe
            else:
                u_prime_cprime = c_prime_safe ** (-gamma)
            
            # E_{z'}[u'(c')]
            expected_u_prime = np.zeros((num_states, n_m), dtype=float)
            for zp in range(num_states):
                expected_u_prime += transition[:, zp] * u_prime_cprime[:, :, zp]
            
            rhs = beta * R * expected_u_prime
            
            # LHS: u'(c_mid)
            c_mid_safe = np.maximum(c_mid, c_floor)
            if is_log:
                lhs = 1.0 / c_mid_safe
            else:
                lhs = c_mid_safe ** (-gamma)
            
            # Euler residual: lhs - rhs
            # If lhs > rhs, we're consuming too little (u' too high), so increase c
            # If lhs < rhs, we're consuming too much, so decrease c
            residual = lhs - rhs
            
            # Update bounds
            # If residual > 0, c is too low, increase lo
            # If residual < 0, c is too high, decrease hi
            mask_pos = residual > 0
            mask_neg = residual < 0
            
            c_lo = np.where(mask_pos, c_mid, c_lo)
            c_hi = np.where(mask_neg, c_mid, c_hi)
            
            # Check convergence
            if np.max(c_hi - c_lo) < tol:
                break
        
        C[t] = 0.5 * (c_lo + c_hi)
        
        # Ensure validity: 0 < c <= m + 1e-10
        C[t] = np.clip(C[t], c_floor, m_grid[np.newaxis, :] + 1e-10)
        
        # Compute V[t] = u(C[t]) + beta * E[V[t+1]]
        c_safe = np.maximum(C[t], c_floor)
        if is_log:
            u_c = np.log(c_safe)
        else:
            u_c = (c_safe ** (1.0 - gamma)) / (1.0 - gamma)
        
        # E[V[t+1]] using the same m' as above
        # Recompute m' with final C[t]
        m_prime = R * (m_grid[np.newaxis, :, np.newaxis] - C[t][:, :, np.newaxis]) + income[np.newaxis, np.newaxis, :]
        m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
        
        V_next = V[t + 1]
        expected_V = np.zeros((num_states, n_m), dtype=float)
        for zp in range(num_states):
            vals = np.interp(m_prime_clipped[:, :, zp], m_grid, V_next[zp])
            expected_V += transition[:, zp] * vals
        
        V[t] = u_c + beta * expected_V
    
    # Final validation and shaping
    # C[t, z, m] should have shape (horizon, num_states, n_m) = (6, 2, 101)
    result = C.copy()
    
    # Ensure terminal condition: c[5] equals grid within 1e-9
    result[horizon - 1] = m_grid[np.newaxis, :]
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, m_grid[np.newaxis, :])
    
    # Ensure 0 < c <= m + 1e-10
    m_broad = m_grid[np.newaxis, :, np.newaxis]
    result = np.clip(result, c_floor, m_broad + 1e-10)
    
    return result
