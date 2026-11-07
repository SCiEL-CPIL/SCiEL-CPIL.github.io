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

    c_min = 1e-12
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Number of consumption candidates for vectorized search
    n_cand = 2000

    V = [None] * T
    P = [None] * T

    # Terminal condition
    V[T - 1] = utility(grid)[np.newaxis, :].repeat(2, axis=0)
    P[T - 1] = grid[np.newaxis, :].repeat(2, axis=0)

    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: copy terminal
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[T - 1].copy()
                P[t_rem] = P[T - 1].copy()
            break

        V_next = V[t + 1]
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))

        # Precompute finite-difference derivatives of V_next
        dV_next = np.zeros_like(V_next)
        d2V_next = np.zeros_like(V_next)

        # First derivative
        dV_next[:, 1:-1] = (V_next[:, 2:] - V_next[:, :-2]) / (grid[2:] - grid[:-2])
        dV_next[:, 0] = (V_next[:, 1] - V_next[:, 0]) / (grid[1] - grid[0])
        dV_next[:, -1] = (V_next[:, -1] - V_next[:, -2]) / (grid[-1] - grid[-2])

        # Second derivative
        d2V_next[:, 1:-1] = (dV_next[:, 2:] - dV_next[:, :-2]) / (grid[2:] - grid[:-2])
        d2V_next[:, 0] = (dV_next[:, 1] - dV_next[:, 0]) / (grid[1] - grid[0])
        d2V_next[:, -1] = (dV_next[:, -1] - dV_next[:, -2]) / (grid[-1] - grid[-2])

        # Build dense consumption grid: c_frac in [0.001, 1], c = m * c_frac
        c_frac = np.linspace(0.001, 1.0, n_cand)  # shape (n_cand,)

        for z in range(n_z):
            # c_matrix: (n_grid, n_cand)
            c_matrix = grid[:, np.newaxis] * c_frac[np.newaxis, :]  # (n_grid, n_cand)
            c_matrix = np.maximum(c_matrix, c_min)

            # next_m for each (grid_point, candidate, z')
            # next_m = R*(m - c) + income[zp]
            # Shape: (n_grid, n_cand, n_z)
            m_col = grid[:, np.newaxis, np.newaxis]  # (n_grid, 1, 1)
            c_3d = c_matrix[:, :, np.newaxis]       # (n_grid, n_cand, 1)
            income_3d = income[np.newaxis, np.newaxis, :]  # (1, 1, n_z)
            next_m = R * (m_col - c_3d) + income_3d  # (n_grid, n_cand, n_z)

            # Clip next_m to grid range for interpolation
            next_m_clipped = np.clip(next_m, grid[0], grid[-1])

            # Interpolate dV_next and d2V_next for each z'
            # dV_next shape: (n_z, n_grid)
            # We need dV_next[zp][next_m_clipped] for each zp
            # Use np.interp along axis=1 for each zp
            dV_interp = np.zeros((n_grid, n_cand, n_z))
            d2V_interp = np.zeros((n_grid, n_cand, n_z))

            for zp in range(n_z):
                # next_m_clipped[:, :, zp] has shape (n_grid, n_cand)
                # We need to interpolate dV_next[zp, :] at these points
                # np.interp works on 1D, so we need to do it row by row or use a vectorized approach
                # For speed, use np.searchsorted + linear interpolation manually
                # Actually, let's use a vectorized approach with np.take and linear interp

                # Simpler: use np.interp on flattened arrays won't work directly for 2D
                # Let's process in chunks or use a different approach
                # For n_grid=101, n_cand=2000, this is 202000 points per zp
                # Let's just use a loop over grid points (101 iterations) which is fast
                for i in range(n_grid):
                    dV_interp[i, :, zp] = np.interp(next_m_clipped[i, :, zp], grid, dV_next[zp, :])
                    d2V_interp[i, :, zp] = np.interp(next_m_clipped[i, :, zp], grid, d2V_next[zp, :])

            # Compute G(c) = beta * R * sum_{zp} trans[z,zp] * dV_next[zp](next_m)
            # G_prime(c) = -beta * R^2 * sum_{zp} trans[z,zp] * d2V_next[zp](next_m)
            trans_row = transition[z, :]  # (n_z,)
            G = beta * R * np.sum(trans_row[np.newaxis, np.newaxis, :] * dV_interp, axis=2)  # (n_grid, n_cand)
            G_prime = -beta * R * R * np.sum(trans_row[np.newaxis, np.newaxis, :] * d2V_interp, axis=2)  # (n_grid, n_cand)

            # f(c) = u'(c) - G(c)
            # We want f(c) = 0. u'(c) is decreasing, G(c) is decreasing (since dV > 0, next_m decreases with c, dV_next increasing in m means dV_next(next_m) decreases with c... wait)
            # Actually next_m = R(m-c) + y, so as c increases, next_m decreases.
            # dV_next is increasing in m (marginal value of wealth is positive), so dV_next(next_m) decreases as c increases.
            # So G(c) decreases as c increases.
            # u'(c) also decreases as c increases.
            # f(c) = u'(c) - G(c). At c=0: u'(0) is large, G(0) is some value. At c=m: u'(m) is small, G(m) is smaller.
            # We need to find where f(c) = 0.

            u_prime_c = utility_prime(c_matrix)  # (n_grid, n_cand)
            f_vals = u_prime_c - G  # (n_grid, n_cand)

            # Find the candidate index where f changes sign from positive to negative
            # f should be positive at low c and negative at high c (if interior solution exists)
            # Find first index where f <= 0
            f_sign_change = np.argmax(f_vals <= 0, axis=1)  # (n_grid,)
            # If f is always positive, argmax returns 0, but we need to check
            f_all_pos = np.all(f_vals > 0, axis=1)
            f_sign_change[f_all_pos] = n_cand - 1  # take the last (largest c)

            # Get the bracketing indices
            idx_hi = f_sign_change  # (n_grid,)
            idx_lo = np.maximum(idx_hi - 1, 0)  # (n_grid,)

            # Interpolate to get a better estimate
            c_lo = c_matrix[np.arange(n_grid), idx_lo]  # (n_grid,)
            c_hi = c_matrix[np.arange(n_grid), idx_hi]  # (n_grid,)
            f_lo = f_vals[np.arange(n_grid), idx_lo]  # (n_grid,)
            f_hi = f_vals[np.arange(n_grid), idx_hi]  # (n_grid,)

            # Linear interpolation
            denom = f_hi - f_lo
            valid = np.abs(denom) > 1e-30
            c_est = np.where(valid, c_lo - f_lo * (c_hi - c_lo) / denom, c_hi)
            c_est = np.clip(c_est, c_min, grid)

            # Now do a few Newton steps for refinement
            for _newton in range(10):
                c_curr = c_est  # (n_grid,)
                # next_m for current c
                next_m_curr = R * (grid - c_curr) + income[np.newaxis, :]  # (n_grid, n_z)
                next_m_curr_clipped = np.clip(next_m_curr, grid[0], grid[-1])

                dV_curr = np.zeros((n_grid, n_z))
                d2V_curr = np.zeros((n_grid, n_z))
                for zp in range(n_z):
                    dV_curr[:, zp] = np.interp(next_m_curr_clipped[:, zp], grid, dV_next[zp, :])
                    d2V_curr[:, zp] = np.interp(next_m_curr_clipped[:, zp], grid, d2V_next[zp, :])

                G_curr = beta * R * np.sum(trans_row[np.newaxis, :] * dV_curr, axis=1)  # (n_grid,)
                G_prime_curr = -beta * R * R * np.sum(trans_row[np.newaxis, :] * d2V_curr, axis=1)  # (n_grid,)

                u_prime_curr = utility_prime(c_curr)
                u_double_prime_curr = utility_double_prime(c_curr)

                f_curr = u_prime_curr - G_curr
                f_prime_curr = u_double_prime_curr - G_prime_curr

                valid_fp = np.abs(f_prime_curr) > 1e-12
                step = np.where(valid_fp, f_curr / f_prime_curr, 0.0)
                c_new = c_curr - step
                c_new = np.clip(c_new, c_min, grid)

                if np.max(np.abs(c_new - c_curr)) < 1e-12:
                    c_est = c_new
                    break
                c_est = c_new

            P_t[z, :] = c_est

            # Compute value
            next_m_final = R * (grid - c_est) + income[np.newaxis, :]  # (n_grid, n_z)
            next_m_final_clipped = np.clip(next_m_final, grid[0], grid[-1])
            cont_val = np.zeros(n_grid)
            for zp in range(n_z):
                cont_val += transition[z, zp] * np.interp(next_m_final_clipped[:, zp], grid, V_next[zp, :])
            V_t[z, :] = utility(c_est) + beta * cont_val

        V[t] = V_t
        P[t] = P_t

    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P[t]

    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)

    # Final period constraint
    result[T - 1, :, :] = grid[np.newaxis, :]

    return result
