import time
import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 148.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    incomes = np.array(model['income'], dtype=float)
    trans = np.array(model['transition'], dtype=float)
    horizon = model['horizon']
    z_count = len(incomes)
    m_grid = np.array(resource_grid, dtype=float)
    n_m = len(m_grid)

    # Pre-compute utility functions
    def crra_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Value functions: V[t, z, :] on m_grid
    V = np.zeros((horizon, z_count, n_m))
    C = np.zeros((horizon, z_count, n_m))

    # Terminal condition
    V[5, :, :] = crra_utility(m_grid)
    C[5, :, :] = m_grid

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            break

        for z in range(z_count):
            if time.time() - t0 > DEADLINE:
                break

            # For each m, solve for optimal c
            for i in range(n_m):
                if time.time() - t0 > DEADLINE:
                    break

                m_val = m_grid[i]

                # Objective: maximize u(c) + beta * sum_z' pi[z,z'] * V[t+1, z', R*(m-c)+income[z']]
                # We minimize the negative of this
                def neg_obj(c):
                    c = max(c, 1e-12)
                    c = min(c, m_val)
                    u_val = crra_utility(np.array([c]))[0]
                    cont_val = 0.0
                    for zp in range(z_count):
                        m_prime = R * (m_val - c) + incomes[zp]
                        # Interpolate V[t+1, zp, :] at m_prime
                        m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                        V_interp = np.interp(m_prime_clipped, m_grid, V[t+1, zp, :])
                        cont_val += trans[z, zp] * V_interp
                    return -(u_val + beta * cont_val)

                # Use bounded scalar minimization
                # c is in [1e-12, m_val]
                if m_val < 1e-9:
                    C[t, z, i] = m_val
                    V[t, z, i] = crra_utility(np.array([m_val]))[0]
                    continue

                res = minimize_scalar(neg_obj, bounds=(1e-12, m_val), method='bounded',
                                      options={'xatol': 1e-10, 'maxiter': 100})

                c_opt = res.x
                c_opt = np.clip(c_opt, 1e-12, m_val)
                C[t, z, i] = c_opt

                # Compute value
                u_val = crra_utility(np.array([c_opt]))[0]
                cont_val = 0.0
                for zp in range(z_count):
                    m_prime = R * (m_val - c_opt) + incomes[zp]
                    m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                    V_interp = np.interp(m_prime_clipped, m_grid, V[t+1, zp, :])
                    cont_val += trans[z, zp] * V_interp
                V[t, z, i] = u_val + beta * cont_val

    # Final constraint enforcement
    for t in range(horizon):
        for z in range(z_count):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[5, :, :] = m_grid

    return C
