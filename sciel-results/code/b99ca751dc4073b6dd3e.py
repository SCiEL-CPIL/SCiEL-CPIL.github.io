import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_states = len(income)
    n_grid = len(resource_grid)
    m_grid = resource_grid

    # Utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c**(1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c**(-gamma)

    # Initialize value functions and consumption policies
    # V[t, z, m_index] - value function
    # C[t, z, m_index] - consumption policy
    V = np.zeros((horizon + 1, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Terminal condition: at t=horizon, consume all (but we only store up to horizon-1 for C)
    # Actually, the problem says c[5] (last period, index 5 since horizon=6) equals grid.
    # So at t=horizon-1, c = m.
    for z in range(n_states):
        C[horizon - 1, z, :] = m_grid
        V[horizon, z, :] = u(m_grid)

    # Precompute interpolation helper
    def interp_v(v_func, m_val):
        """Interpolate value function at m_val"""
        if m_val <= m_grid[0]:
            return v_func[0]
        elif m_val >= m_grid[-1]:
            return v_func[-1]
        else:
            return np.interp(m_val, m_grid, v_func)

    # Work backward from t = horizon - 2 to t = 0
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # For each m, find optimal c using FOC + bisection
            for i, m in enumerate(m_grid):
                # FOC: u'(c) = beta * R * sum_z' P[z,z'] * V'[t+1, z', R*(m-c) + income[z']]
                # We need V'[t+1, z', m'] which we approximate via finite differences
                
                # Define the objective function to maximize
                def objective(c):
                    if c <= 0 or c > m + 1e-10:
                        return -np.inf
                    m_next = R * (m - c)
                    cont_val = 0.0
                    for zp in range(n_states):
                        next_m = m_next + income[zp]
                        v_next = interp_v(V[t + 1, zp, :], next_m)
                        cont_val += transition[z, zp] * v_next
                    return u(c) + beta * cont_val

                # Use a combination of grid search and local refinement
                # First, coarse grid search
                n_coarse = 50
                c_coarse = m * np.linspace(0.001, 1.0, n_coarse)
                c_coarse = np.maximum(c_coarse, 1e-12)
                c_coarse = np.minimum(c_coarse, m)
                
                obj_coarse = np.array([objective(c) for c in c_coarse])
                best_idx = np.argmax(obj_coarse)
                
                # Local refinement around best
                if best_idx > 0 and best_idx < n_coarse - 1:
                    c_left = c_coarse[best_idx - 1]
                    c_right = c_coarse[best_idx + 1]
                elif best_idx == 0:
                    c_left = c_coarse[0]
                    c_right = c_coarse[1]
                else:
                    c_left = c_coarse[best_idx - 1]
                    c_right = c_coarse[best_idx]
                
                # Golden section search in [c_left, c_right]
                def golden_search(f, a, b, tol=1e-10, max_iter=100):
                    invphi = (np.sqrt(5.0) - 1.0) / 2.0
                    invphi2 = invphi * invphi
                    c = b - invphi * (b - a)
                    d = a + invphi * (b - a)
                    fc = f(c)
                    fd = f(d)
                    for _ in range(max_iter):
                        if abs(b - a) < tol:
                            break
                        if fc > fd:
                            b = d
                            d = c
                            fd = fc
                            c = b - invphi * (b - a)
                            fc = f(c)
                        else:
                            a = c
                            c = d
                            fc = fd
                            d = a + invphi * (b - a)
                            fd = f(d)
                    return 0.5 * (a + b)
                
                best_c = golden_search(objective, c_left, c_right)
                best_c = np.clip(best_c, 1e-12, m)
                
                C[t, z, i] = best_c
                V[t, z, i] = objective(best_c)

        # Check time
        if time.time() > deadline - 5.0:
            break

    # Enforce terminal condition: at t=horizon-1, c = m
    for z in range(n_states):
        C[horizon - 1, z, :] = m_grid

    # Validate and fix any issues
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                if C[t, z, i] <= 0:
                    C[t, z, i] = 1e-12
                if C[t, z, i] > m_grid[i] + 1e-10:
                    C[t, z, i] = m_grid[i]

    return C
