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

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Backward induction
    # Exploit homotheticity: search over f = c/m
    # Use FOC-based root finding with fallback to grid search
    
    # Number of points for coarse search
    coarse_n = 80

    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    C[t2, z, :] = np.clip(grid * 0.4, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            for i_m in range(n_grid):
                m = grid[i_m]
                
                if time.time() > deadline - 1.0:
                    C[t, z, i_m] = np.clip(m * 0.4, 1e-12, m)
                    V[t, z, i_m] = utility(C[t, z, i_m])
                    continue

                # Coarse grid search to find a good region
                f_coarse = np.linspace(0.001, 1.0, coarse_n)
                best_f = f_coarse[0]
                best_val = -np.inf
                
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
                    if total > best_val:
                        best_val = total
                        best_f = f
                
                # Local refinement with minimize_scalar
                f_lower = max(0.001, best_f - 0.05)
                f_upper = min(1.0, best_f + 0.05)
                
                def neg_objective(f):
                    c = f * m
                    if c <= 0:
                        return 0.0
                    c = max(c, 1e-12)
                    
                    expected_cont = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                        expected_cont += transition[z, zp] * cont_val
                    
                    total = utility(np.array([c]))[0] + beta * expected_cont
                    return -total

                try:
                    result = minimize_scalar(neg_objective, bounds=(f_lower, f_upper), 
                                             method='bounded', options={'xatol': 1e-9})
                    f_opt = result.x
                except:
                    f_opt = best_f
                
                f_opt = np.clip(f_opt, 0.001, 1.0)
                c_opt = f_opt * m
                c_opt = np.clip(c_opt, 1e-12, m + 1e-10)
                
                C[t, z, i_m] = c_opt
                
                expected_cont = 0.0
                for zp in range(n_states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                    expected_cont += transition[z, zp] * cont_val
                
                V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont

    # Ensure validity
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
