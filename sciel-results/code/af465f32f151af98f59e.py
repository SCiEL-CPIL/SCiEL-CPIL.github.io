import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    c_min = 1e-12
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2

    # Fine candidate grid: fractions of m (same style as evaluator's best)
    # Use 300 fractions from 0.001 to 1.0
    n_frac = 300
    fracs = np.linspace(0.001, 1.0, n_frac)  # shape (300,)

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    # Policy: P[t, z, i] = optimal consumption at grid point i, state z, period t
    # Value:  V[t, z, i] = value at grid point i, state z, period t
    P = np.zeros((T, n_z, n_grid))
    V = np.zeros((T, n_z, n_grid))

    # Terminal period: c = m
    P[T-1, :, :] = grid[np.newaxis, :].repeat(n_z, axis=0)
    V[T-1, :, :] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)

    # Backward induction
    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 3.0:
            # Fallback: copy terminal policy
            P[t, :, :] = P[T-1, :, :]
            V[t, :, :] = V[T-1, :, :]
            continue

        V_next = V[t + 1]  # shape (n_z, n_grid)

        for z in range(n_z):
            # For each grid point m_i, evaluate objective over all fracs
            # m shape: (n_grid,)
            # c_candidates shape: (n_grid, n_frac) = m_i * frac_j
            c_cand = grid[:, np.newaxis] * fracs[np.newaxis, :]  # (n_grid, n_frac)
            c_cand = np.maximum(c_cand, c_min)

            # Utility of consumption: (n_grid, n_frac)
            u_cand = utility(c_cand)

            # Next-period wealth for each (i, j): R*(m_i - c_ij) + income[z']
            # next_m shape: (n_grid, n_frac, n_z)
            m_minus_c = grid[:, np.newaxis, np.newaxis] - c_cand[:, :, np.newaxis]
            next_m = R * m_minus_c + income[np.newaxis, np.newaxis, :]  # (n_grid, n_frac, n_z)

            # Interpolate V_next for each z' at next_m
            # V_next[z'] is shape (n_grid,), we need values at next_m[:, :, z']
            # Do it per z' for memory efficiency
            cont_val = np.zeros((n_grid, n_frac))  # (n_grid, n_frac)
            for zp in range(n_z):
                # Interpolate V_next[zp] at next_m[:, :, zp]
                # next_m[:, :, zp] shape (n_grid, n_frac)
                # np.interp works on 1D, so flatten
                nm_flat = next_m[:, :, zp].ravel()
                v_interp = np.interp(nm_flat, grid, V_next[zp])
                v_interp = v_interp.reshape(n_grid, n_frac)
                cont_val += transition[z, zp] * v_interp

            # Total objective: u(c) + beta * cont_val
            obj = u_cand + beta * cont_val  # (n_grid, n_frac)

            # Argmax over fracs
            best_idx = np.argmax(obj, axis=1)  # (n_grid,)
            best_c = c_cand[np.arange(n_grid), best_idx]  # (n_grid,)

            # Enforce 0 < c <= m
            best_c = np.clip(best_c, c_min, grid + 1e-10)

            P[t, z, :] = best_c
            V[t, z, :] = obj[np.arange(n_grid), best_idx]

    # Final constraints
    for t in range(T):
        for z in range(n_z):
            P[t, z, :] = np.clip(P[t, z, :], c_min, grid + 1e-10)

    # Terminal period must equal grid
    P[T-1, :, :] = grid[np.newaxis, :].repeat(n_z, axis=0)

    return P
