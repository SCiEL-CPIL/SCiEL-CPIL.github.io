import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)

    # Use a fine grid for the value function
    m_fine = np.geomspace(0.001, 40, 401)
    n_fine = len(m_fine)

    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))

    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)

    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            V_t_z = np.zeros(n_fine)
            
            # Vectorized computation for all m values at once
            for i in range(n_fine):
                m_val = m_fine[i]
                
                # Grid of consumption fractions
                n_c_grid = 200
                c_fracs = np.linspace(0.001, 1.0, n_c_grid)
                c_vals = m_val * c_fracs
                c_vals = np.maximum(c_vals, 1e-12)
                
                # Compute continuation value
                cont_val = np.zeros(len(c_vals))
                for zp in range(Z):
                    m_prime = R * (m_val - c_vals) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    v_interp = np.interp(m_prime, m_fine, V[t+1, zp, :])
                    cont_val += transition[z, zp] * v_interp
                
                obj = utility(c_vals) + beta * cont_val
                idx = np.argmax(obj)
                V_t_z[i] = obj[idx]
            
            V[t, z, :] = V_t_z
        
        # Check time
        if time.time() > deadline - 15:
            break

    # Now extract the consumption policy
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period: consume everything
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, use grid search with local refinement
    for t in range(horizon - 1):
        for z in range(Z):
            for i in range(M):
                m_val = m_grid[i]
                
                # Coarse grid search
                n_c_coarse = 150
                c_fracs = np.linspace(0.001, 1.0, n_c_coarse)
                c_vals = m_val * c_fracs
                c_vals = np.maximum(c_vals, 1e-12)
                
                cont_val = np.zeros(len(c_vals))
                for zp in range(Z):
                    m_prime = R * (m_val - c_vals) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    v_interp = np.interp(m_prime, m_fine, V[t+1, zp, :])
                    cont_val += transition[z, zp] * v_interp
                
                obj = utility(c_vals) + beta * cont_val
                idx = np.argmax(obj)
                c = c_vals[idx]
                
                # Local refinement around the coarse solution
                c_low = max(1e-12, c * 0.85)
                c_high = min(m_val, c * 1.15)
                n_c_fine = 100
                c_local = np.linspace(c_low, c_high, n_c_fine)
                c_local = np.clip(c_local, 1e-12, m_val)
                
                cont_val_local = np.zeros(len(c_local))
                for zp in range(Z):
                    m_prime = R * (m_val - c_local) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    v_interp = np.interp(m_prime, m_fine, V[t+1, zp, :])
                    cont_val_local += transition[z, zp] * v_interp
                
                obj_local = utility(c_local) + beta * cont_val_local
                idx_local = np.argmax(obj_local)
                consumption[t, z, i] = c_local[idx_local]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
