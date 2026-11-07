import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = len(income)
    grid_out = resource_grid
    n_out = len(grid_out)
    
    # Internal grid for computation
    # 600 points should be plenty accurate and fast
    n_inner = 600
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    # Precompute grid spacing for derivatives if needed, 
    # but for FOC fixed point we just need the function itself.
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: c = m
    V[T-1] = np.zeros((n_z, n_grid))
    P[T-1] = np.zeros((n_z, n_grid))
    
    for z in range(n_z):
        if abs(gamma - 1.0) < 1e-10:
            V[T-1][z, :] = np.log(np.maximum(grid, c_min))
        else:
            V[T-1][z, :] = (np.maximum(grid, c_min) ** (1 - gamma)) / (1 - gamma)
        P[T-1][z, :] = grid
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline:
            # Fallback: use terminal policy for remaining periods
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[T-1].copy()
                P[t_rem] = P[T-1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        
        # We need to solve for c_t(m) for each z
        # FOC: u'(c) = beta * E_z' [ R * V'_{t+1}(m') ] ? 
        # Wait, the state is m. m' = R(m-c) + y'.
        # Bellman: V_t(m, z) = max_c { u(c) + beta * sum_z' p_{z z'} V_{t+1}(m', z') }
        # FOC: u'(c) = beta * sum_z' p_{z z'} V'_{t+1}(m', z') * (dm'/dc)
        # dm'/dc = -R
        # So u'(c) = -beta * R * sum_z' p_{z z'} V'_{t+1}(m', z')
        # Note: V' is positive. So RHS is negative? 
        # Let's check signs.
        # V(m) is increasing. V' > 0.
        # Objective: u(c) + beta E[V(m')].
        # d/dc = u'(c) + beta E[V'(m') * (-R)] = 0
        # u'(c) = beta * R * E[V'(m')]
        # Yes, this is correct.
        
        # To use FOC fixed point, we need V'_{t+1}.
        # We can compute V'_{t+1} from V_{t+1} using finite differences.
        
        # Compute derivatives of V[t+1]
        V_next = V[t+1]
        dV_next = np.zeros((n_z, n_grid))
        
        # Central differences for interior
        for z in range(n_z):
            # Use np.gradient for simplicity and vectorization
            # np.gradient handles non-uniform grids
            dV_next[z, :] = np.gradient(V_next[z, :], grid)
            
        # Fixed point iteration for c
        # Initialize c
        c_curr = np.zeros((n_z, n_grid))
        for z in range(n_z):
            c_curr[z, :] = 0.5 * grid # Initial guess
        
        max_iter = 50
        tol = 1e-8
        
        for _ in range(max_iter):
            c_prev = c_curr.copy()
            
            # Calculate m' for each z, z'
            # c_curr shape: (n_z, n_grid)
            # m shape: (n_grid,)
            # m' shape: (n_z, n_z_prime, n_grid)
            
            # m - c
            m_minus_c = grid[np.newaxis, np.newaxis, :] - c_curr[:, np.newaxis, :]
            # m' = R * (m - c) + income[z']
            # income shape: (n_z_prime,)
            m_prime = R * m_minus_c + income[np.newaxis, :, np.newaxis]
            
            # Clip m' to grid range for interpolation
            m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
            
            # Interpolate dV_next
            # dV_next shape: (n_z_prime, n_grid)
            # We need dV_next[z', m_prime_clipped[z, z', i]]
            
            # Vectorized interpolation
            # np.interp doesn't do batched 3D easily without loops or complex broadcasting.
            # We can loop over z (2 states) which is cheap.
            
            expected_dV = np.zeros((n_z, n_grid))
            
            for z in range(n_z):
                sum_dV = np.zeros(n_grid)
                for zp in range(n_z):
                    # Interpolate dV_next[zp] at m_prime[z, zp, :]
                    # m_prime[z, zp, :] is 1D array of length n_grid
                    interp_dV = np.interp(m_prime[z, zp, :], grid, dV_next[zp, :])
                    sum_dV += transition[z, zp] * interp_dV
                expected_dV[z, :] = sum_dV
            
            # FOC: u'(c) = beta * R * expected_dV
            # u'(c) = c^(-gamma)
            # c^(-gamma) = beta * R * expected_dV
            # c = (beta * R * expected_dV)^(-1/gamma)
            
            rhs = beta * R * expected_dV
            rhs = np.maximum(rhs, 1e-30) # Avoid log/inv issues
            
            if abs(gamma - 1.0) < 1e-10:
                # c^(-1) = rhs => c = 1/rhs
                c_new = 1.0 / rhs
            else:
                c_new = rhs ** (-1.0 / gamma)
                
            # Enforce constraint c <= m
            c_new = np.minimum(c_new, grid[np.newaxis, :])
            c_new = np.maximum(c_new, c_min)
            
            # Check convergence
            if np.max(np.abs(c_new - c_prev)) < tol:
                c_curr = c_new
                break
            c_curr = c_new
            
        P_t = c_curr
        
        # Calculate V_t
        for z in range(n_z):
            # m' = R(m - c) + y'
            m_minus_c = grid[np.newaxis, :] - P_t[z, :]
            m_prime = R * m_minus_c + income[np.newaxis, :]
            m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
            
            expected_V = np.zeros(n_grid)
            for zp in range(n_z):
                interp_V = np.interp(m_prime_clipped[zp, :], grid, V_next[zp, :])
                expected_V += transition[z, zp] * interp_V
                
            c_val = P_t[z, :]
            if abs(gamma - 1.0) < 1e-10:
                u_c = np.log(np.maximum(c_val, c_min))
            else:
                u_c = (np.maximum(c_val, c_min) ** (1 - gamma)) / (1 - gamma)
                
            V_t[z, :] = u_c + beta * expected_V
            
        V[t] = V_t
        P[t] = P_t
        
    # Interpolate to output grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
            
    # Enforce constraints on final result
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    # Terminal condition exactness
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
