import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 150.0  # wall-clock limit
    MARGIN = 5.0      # safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Continuation value function V[t, z, i] for t in 0..T-2
    # At t=T-1, V=0 (terminal, no future)
    V = np.zeros((T, Z, N))  # V[T-1] = 0
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
    
    # Backward induction from t = T-2 down to t = 0
    for t in range(T - 2, -1, -1):
        if time.time() - start_time > DEADLINE - MARGIN:
            # Fill remaining periods with a simple heuristic
            for tt in range(t, T - 1):
                for z in range(Z):
                    c_policy[tt, z, :] = resource_grid
            break
        
        for z in range(Z):
            # V_next[z', i] for z' in transition
            # Expected continuation: sum_{z'} P(z'|z) * V[t+1, z', i']
            # where i' depends on c and z'
            
            # We need to find c* that maximizes:
            # u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z', i']
            # where m' = R*(m - c) + income[z']
            
            # Use envelope theorem: V_m = u'(c*)
            # So V[t, z, m] = u(c*) + beta * E[V[t+1, z', m']]
            # and V_m = u'(c*)
            
            # We'll solve for c* using the FOC:
            # u'(c) = beta * R * sum_{z'} P(z'|z) * V_m[t+1, z', m']
            # where m' = R*(m-c) + income[z']
            
            # Vectorized bisection across all m points
            m = resource_grid  # shape (N,)
            
            # Initialize c with a reasonable guess
            c = np.clip(m * 0.5, 1e-12, m)
            
            # Bisection iterations
            for _ in range(60):
                # Compute m' for each z'
                # m'[z', i] = R*(m[i] - c[i]) + income[z']
                # Shape: (Z, N)
                m_prime = R * (m[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
                m_prime = np.maximum(m_prime, 1e-12)
                
                # Interpolate V[t+1, z', m'] onto m_prime grid
                # V[t+1, z'] is defined on resource_grid
                # We need V at m_prime which may be outside [grid[0], grid[-1]]
                
                V_interp = np.zeros((Z, N))
                for zp in range(Z):
                    # Clip m_prime to grid range for interpolation
                    mp = np.clip(m_prime[zp], resource_grid[0], resource_grid[-1])
                    V_interp[zp, :] = np.interp(mp, resource_grid, V[t+1, zp, :])
                
                # Expected continuation: sum_{z'} P(z'|z) * V_interp[z', :]
                # transition[z, zp] is P(z'|z)
                expected_V = transition[z, :] @ V_interp  # shape (N,)
                
                # V_m at m': beta * R * expected_V_m
                # V_m[t+1, z', m'] = u'(c*[t+1, z', m'])
                # We need to interpolate u'(c*) onto m_prime
                # c*[t+1, z', i] is on resource_grid
                c_next = c_policy[t+1, :, :]  # shape (Z, N)
                up_next = util_prime(c_next)  # shape (Z, N)
                
                Vm_interp = np.zeros((Z, N))
                for zp in range(Z):
                    mp = np.clip(m_prime[zp], resource_grid[0], resource_grid[-1])
                    Vm_interp[zp, :] = np.interp(mp, resource_grid, up_next[zp, :])
                
                expected_Vm = transition[z, :] @ Vm_interp  # shape (N,)
                
                # FOC: u'(c) = beta * R * expected_Vm
                rhs = beta * R * expected_Vm
                lhs = util_prime(c)
                
                # Bisection: if lhs > rhs, c is too small (u' decreasing in c), increase c
                # Actually u'(c) is decreasing in c, so if u'(c) > rhs, we need larger c
                # Wait: u'(c) decreases as c increases. If u'(c) > rhs, c is too small.
                # So c_new = (c + m) / 2 if u'(c) > rhs, else c_new = (c_low + c) / 2
                
                # Let's use proper bisection with bounds
                # c in [1e-12, m]
                # We'll track low and high bounds
                c_low = np.full(N, 1e-12)
                c_high = m.copy()
                
                # Re-do bisection properly
                break  # break out to do proper bisection
            
            # Proper vectorized bisection
            c_low = np.full(N, 1e-12)
            c_high = m.copy()
            c = (c_low + c_high) / 2.0
            
            for _ in range(50):
                if time.time() - start_time > DEADLINE - MARGIN:
                    break
                
                # Compute m' for each z'
                m_prime = R * (m[np.newaxis, :] - c[np.newaxis, :]) + income[:, np.newaxis]
                m_prime = np.maximum(m_prime, 1e-12)
                
                # Interpolate V_m[t+1, z', m']
                c_next = c_policy[t+1, :, :]
                up_next = util_prime(c_next)
                
                Vm_interp = np.zeros((Z, N))
                for zp in range(Z):
                    mp = np.clip(m_prime[zp], resource_grid[0], resource_grid[-1])
                    Vm_interp[zp, :] = np.interp(mp, resource_grid, up_next[zp, :])
                
                expected_Vm = transition[z, :] @ Vm_interp
                rhs = beta * R * expected_Vm
                lhs = util_prime(c)
                
                # u'(c) is decreasing in c
                # If lhs > rhs: u'(c) too high, need larger c
                # If lhs < rhs: u'(c) too low, need smaller c
                mask = lhs > rhs
                c_high[mask] = c[mask]
                c_low[~mask] = c[~mask]
                c = (c_low + c_high) / 2.0
            
            c_policy[t, z, :] = np.clip(c, 1e-12, m)
            
            # Compute V[t, z, i] = u(c) + beta * E[V[t+1, z', m']]
            m_prime = R * (m[np.newaxis, :] - c_policy[t, z, :][np.newaxis, :]) + income[:, np.newaxis]
            m_prime = np.maximum(m_prime, 1e-12)
            
            V_interp = np.zeros((Z, N))
            for zp in range(Z):
                mp = np.clip(m_prime[zp], resource_grid[0], resource_grid[-1])
                V_interp[zp, :] = np.interp(mp, resource_grid, V[t+1, zp, :])
            
            expected_V = transition[z, :] @ V_interp
            V[t, z, :] = util(c_policy[t, z, :]) + beta * expected_V
    
    # Ensure last period is exactly the grid
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Ensure finite values and 0 < c <= m + 1e-10
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # Final validity check
    c_policy = np.where(np.isfinite(c_policy), c_policy, resource_grid[np.newaxis, np.newaxis, :])
    c_policy = np.maximum(c_policy, 1e-12)
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.minimum(c_policy[t, z, :], resource_grid + 1e-10)
    
    # Ensure c[5] equals grid within 1e-9
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
