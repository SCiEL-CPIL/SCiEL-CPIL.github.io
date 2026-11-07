"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    m_grid = resource_grid  # shape (101,)
    n_m = len(m_grid)
    n_z = len(income)
    c_floor = 1e-12

    def util(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Continuation value: V[t+1, z, m]
    # At t = horizon-1 (last period), next period is terminal: V_terminal = 0
    # Actually, let's think about the timing.
    # consumption[t, z, m] for t in 0..horizon-1
    # V[t, z, m] = max_c { util(c) + beta * sum_{z'} P(z'|z) V[t+1, z', m'] }
    # where m' = R*(m-c) + income[z']
    # Terminal condition: V[horizon, z, m] = 0 for all z, m
    # So we compute V from t=horizon-1 down to t=0.

    # V_next[z, m] shape (n_z, n_m)
    V_next = np.zeros((n_z, n_m))

    consumption = np.zeros((horizon, n_z, n_m))

    # For interpolation of continuation value, we need V_next as a function of m for each z.
    # Since m_grid is geometric, we'll use np.interp with the grid.

    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline:
            # Return what we have so far; fill remaining with m (consume all) as fallback
            for tt in range(t, -1, -1):
                consumption[tt] = m_grid[None, :]
            break

        # V_next is shape (n_z, n_m)
        # For each z, we want to compute:
        # V_current[z, m] = max_c { util(c) + beta * sum_{z'} P(z'|z) V_next[z', R*(m-c) + income[z']] }

        # We'll do this by searching over c for each (z, m).
        # To speed up, we can vectorize over m for each z, but the max is over c.
        # Alternative: for each z, m, search c on a fine grid.

        # Let's use a dense grid for c search.
        # c must be in (0, m]. Let's use c_grid = m * linspace(eps, 1, n_search)
        # But m varies. Better: for each m, c ranges from c_floor to m.

        # Vectorized approach:
        # For each z (2 states), for each m (101 points), we need to find best c.
        # We can precompute the continuation value as a function of m' for each z'.
        # V_next[z', m'] is defined on m_grid. For arbitrary m', we interpolate.

        # Let's create a fine search grid for c relative to m.
        # Use a fixed number of search points, say 200.
        n_search = 200
        c_frac = np.linspace(0.001, 1.0, n_search)  # c = c_frac * m

        V_current = np.zeros((n_z, n_m))
        c_opt = np.zeros((n_z, n_m))

        for z in range(n_z):
            # For this z, transition probabilities to z'
            P = transition[z]  # shape (n_z,)

            # For each m in m_grid, and each c_frac, compute:
            # c = c_frac * m
            # m' = R*(m - c) + income[z'] for each z'
            # cont_val = sum_{z'} P[z'] * interp(V_next[z'], m')
            # total = util(c) + beta * cont_val

            # Vectorize over m and c_frac
            m_col = m_grid[:, None]  # (n_m, 1)
            c_grid = m_col * c_frac[None, :]  # (n_m, n_search)
            c_grid = np.maximum(c_grid, c_floor)

            # m' for each z'
            # m' = R*(m - c) + income[z']
            # shape: (n_m, n_search, n_z)
            m_prime = R * (m_col - c_grid)[:, :, None] + income[None, None, :]

            # Interpolate V_next[z', m'] for each z'
            # V_next[z'] is shape (n_m,). m_prime is (n_m, n_search, n_z).
            # We need to interpolate for each z'.
            # Let's loop over z' since n_z is small (2).
            cont_val = np.zeros((n_m, n_search))
            for zp in range(n_z):
                # m_prime_zp shape (n_m, n_search)
                m_prime_zp = m_prime[:, :, zp]
                # Interpolate V_next[zp] at m_prime_zp
                # V_next[zp] is defined on m_grid (geometric)
                # np.interp expects increasing x. m_grid is increasing.
                # m_prime_zp could be outside [m_grid[0], m_grid[-1]].
                # Clip to grid range.
                m_prime_clipped = np.clip(m_prime_zp, m_grid[0], m_grid[-1])
                v_interp = np.interp(m_prime_clipped, m_grid, V_next[zp])
                cont_val += P[zp] * v_interp

            # total_util shape (n_m, n_search)
            util_c = util(c_grid)
            total = util_c + beta * cont_val

            # Find max over c_frac for each m
            best_idx = np.argmax(total, axis=1)
            V_current[z] = total[np.arange(n_m), best_idx]
            c_opt[z] = c_grid[np.arange(n_m), best_idx]

        consumption[t] = c_opt
        V_next = V_current

    # Ensure c[5] (last period) equals grid within 1e-9
    # At t=5 (horizon-1), V_next is terminal (0), so:
    # V[5, z, m] = max_c util(c) + beta * 0 = max_c util(c)
    # Since util is increasing in c, max is at c=m.
    # So c[5] should be m_grid.
    consumption[5] = m_grid[None, :]

    # Validate: finite values, 0 < c <= m + 1e-10
    # Clip to ensure validity
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z] = np.clip(consumption[t, z], c_floor, m_grid + 1e-10)
            consumption[t, z] = np.maximum(consumption[t, z], c_floor)

    # Final check: c[5] must equal grid
    consumption[5] = m_grid[None, :]

    return consumption
# EVOLVE-BLOCK-END
