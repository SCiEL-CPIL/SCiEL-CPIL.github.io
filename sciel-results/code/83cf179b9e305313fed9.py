"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Continuation values: V[t, z, m] = max expected utility from t to horizon
    # At t = horizon-1 (last period, index 5), consume everything: c = m
    # So V[5, z, m] = utility(m)
    
    # We'll compute value functions backward
    # V[t, z] as a function of m (resource)
    
    # For interpolation, we'll store V on the resource_grid
    # V shape: (horizon, n_states, n_grid)
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))
    
    # Terminal: consume all
    V[horizon - 1, :, :] = utility(resource_grid)
    C[horizon - 1, :, :] = resource_grid
    
    # Backward induction
    # For each t from horizon-2 down to 0:
    #   For each state z:
    #     For each m in resource_grid:
    #       Choose c in [0, m] to maximize:
    #         utility(c) + beta * sum_z' transition[z, z'] * V[t+1, z', R*(m-c) + income[z']]
    
    # To speed up, we'll use a finer search grid for c
    # and interpolate continuation values
    
    # Use a denser grid for c search
    # The budget parameter suggests we use ~budget points for c search
    c_frac_grid = np.linspace(0.001, 1.0, budget)
    
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            # Not enough time for remaining iterations; break
            break
        
        for z in range(n_states):
            # For each m, find optimal c
            best_c = np.zeros(n_grid)
            best_val = np.zeros(n_grid)
            
            for i, m in enumerate(resource_grid):
                # Search over c = m * frac
                c_vals = m * c_frac_grid
                # Ensure c <= m (c_frac_grid max is 1.0, so c <= m)
                c_vals = np.minimum(c_vals, m)
                
                # Next resources: m' = R * (m - c) + income[z']
                # For each z', compute continuation value
                # V[t+1, z', m'] where m' = R*(m-c) + income[z']
                
                # We need to interpolate V[t+1, z', :] at m'
                # m' can be outside resource_grid range, so clip
                m_min = resource_grid[0]
                m_max = resource_grid[-1]
                
                expected_cont = np.zeros(len(c_vals))
                for zp in range(n_states):
                    m_prime = R * (m - c_vals) + income[zp]
                    # Clip to grid range for interpolation
                    m_prime_clipped = np.clip(m_prime, m_min, m_max)
                    # Interpolate
                    cont_vals = np.interp(m_prime_clipped, resource_grid, V[t+1, zp, :])
                    expected_cont += transition[z, zp] * cont_vals
                
                total_vals = utility(c_vals) + beta * expected_cont
                idx = np.argmax(total_vals)
                best_c[i] = c_vals[idx]
                best_val[i] = total_vals[idx]
            
            C[t, z, :] = best_c
            V[t, z, :] = best_val
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            # Fill remaining periods with terminal-like behavior
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    # Simple fallback: consume proportionally
                    C[t2, z, :] = np.clip(resource_grid * 0.5, 1e-12, resource_grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = resource_grid.copy()
    
    return C
# EVOLVE-BLOCK-END
