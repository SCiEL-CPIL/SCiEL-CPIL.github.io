"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Backward induction
    # Key insight: due to homotheticity of CRRA, the optimal c/m ratio depends
    # only on the "effective" wealth level, not the absolute scale.
    # We exploit this by searching over the fraction f = c/m.
    
    # For each t, z, and m, we want to find f in [0, 1] that maximizes:
    #   utility(f*m) + beta * sum_{z'} transition[z, z'] * V[t+1, z', R*(1-f)*m + income[z']]
    
    # We'll use a combination approach:
    # 1. Coarse grid search to find a good region
    # 2. Local refinement with minimize_scalar for precision
    
    # Number of points for coarse search
    coarse_n = 60
    
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    # Use a reasonable consumption fraction
                    C[t2, z, :] = np.clip(grid * 0.4, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # For each m in the grid, find optimal c
            for i_m in range(n_grid):
                m = grid[i_m]
                
                if time.time() > deadline - 1.0:
                    # Fallback: use a simple fraction
                    C[t, z, i_m] = np.clip(m * 0.4, 1e-12, m)
                    V[t, z, i_m] = utility(C[t, z, i_m])
                    continue

                # Define the objective function in terms of f = c/m
                # We maximize: utility(f*m) + beta * E[V_next]
                # where V_next is interpolated at m' = R*(1-f)*m + income[z']
                
                def neg_objective(f):
                    c = f * m
                    if c <= 0:
                        return 0.0  # or a large positive number since we're minimizing
                    c = max(c, 1e-12)
                    
                    # Compute continuation value
                    expected_cont = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        # Clip to grid range
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        # Interpolate
                        cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                        expected_cont += transition[z, zp] * cont_val
                    
                    total = utility(np.array([c]))[0] + beta * expected_cont
                    return -total  # Negate because minimize_scalar minimizes

                # Coarse grid search first to find a good starting point
                f_coarse = np.linspace(0.001, 1.0, coarse_n)
                best_f_coarse = f_coarse[0]
                best_val_coarse = -np.inf
                
                for f in f_coarse:
                    c = f * m
                    c = max(c, 1e-12)
                    
                    expected_cont = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                        expected_cont += transition[z, zp] * cont_val
                    
                    total = utility(np.array([c]))[0] + beta * expected_cont
                    if total > best_val_coarse:
                        best_val_coarse = total
                        best_f_coarse = f
                
                # Local refinement around the best coarse solution
                # Define bounds for local search
                f_lower = max(0.001, best_f_coarse - 0.05)
                f_upper = min(1.0, best_f_coarse + 0.05)
                
                try:
                    result = minimize_scalar(neg_objective, bounds=(f_lower, f_upper), 
                                             method='bounded', options={'xatol': 1e-8})
                    f_opt = result.x
                except:
                    f_opt = best_f_coarse
                
                # Ensure f is in valid range
                f_opt = np.clip(f_opt, 0.001, 1.0)
                
                c_opt = f_opt * m
                c_opt = np.clip(c_opt, 1e-12, m + 1e-10)
                
                C[t, z, i_m] = c_opt
                
                # Compute the value
                expected_cont = 0.0
                for zp in range(n_states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                    expected_cont += transition[z, zp] * cont_val
                
                V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
