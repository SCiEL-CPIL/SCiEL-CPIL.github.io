"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    """Solve consumption-saving problem via backward induction."""
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
    n_z = len(income)  # 2

    # c candidates: m * linspace(0.002, 1, budget)
    c_frac = np.linspace(0.002, 1.0, budget)  # shape (48,)
    # c_candidates[z, m_idx, c_idx] = m_grid[m_idx] * c_frac[c_idx]
    c_candidates = m_grid[None, :, None] * c_frac[None, None, :]  # (n_m, budget)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Value function V[z, m_idx] for current period
    # Start from t = horizon-1 (last period): consume all
    V = np.zeros((n_z, n_m))
    consumption = np.zeros((horizon, n_z, n_m))

    # Last period: c = m (consume everything)
    consumption[horizon - 1] = m_grid[None, :]  # (n_z, n_m)

    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Fill remaining with proportional consumption
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[None, :] * 0.5
            break

        # For each z and m, find optimal c
        # Continuation: m' = R*(m - c) + income[z']
        # E[V[z', m']] = sum_z' transition[z, z'] * V[z', m']
        # V is defined on m_grid, so we need to interpolate m' onto m_grid

        new_V = np.zeros((n_z, n_m))
        new_c = np.zeros((n_z, n_m))

        for z in range(n_z):
            # For each m_idx, compute utility(c) + beta * E[V[z', m']]
            # c_candidates[z, m_idx, :] = m_grid[m_idx] * c_frac
            # m_prime = R * (m_grid[m_idx] - c) + income[z']
            # We need V[z', m'] interpolated

            # Vectorize over m_idx and c_idx
            # c shape: (n_m, budget)
            c_mat = c_candidates  # (n_m, budget)
            util_mat = utility(c_mat)  # (n_m, budget)

            # m_prime[z', m_idx, c_idx] = R*(m_grid[m_idx] - c) + income[z']
            m_prime = R * (m_grid[None, :] - c_mat) + income[None, None, :]
            # m_prime shape: (n_z, n_m, budget)

            # Interpolate V[z', :] onto m_prime
            # V[z'] is defined on m_grid, m_prime may be outside [min, max]
            # Clip m_prime to grid range for interpolation
            m_min, m_max = m_grid[0], m_grid[-1]
            m_prime_clipped = np.clip(m_prime, m_min, m_max)

            # Interpolate: for each z', interpolate V[z'] at m_prime_clipped
            # Use np.interp for each z'
            cont_val = np.zeros((n_z, n_m, budget))
            for zp in range(n_z):
                # V[zp] shape (n_m,), m_prime_clipped[zp] shape (n_m, budget)
                # Flatten and interpolate
                m_flat = m_prime_clipped[zp].ravel()
                v_interp = np.interp(m_flat, m_grid, V[zp])
                cont_val[zp] = v_interp.reshape(n_m, budget)

            # Expected continuation: sum over z' of transition[z, z'] * V[z', m']
            # transition[z] shape (n_z,)
            cont_expected = np.tensordot(transition[z], cont_val, axes=(0, 0))
            # cont_expected shape: (n_m, budget)

            total_util = util_mat + beta * cont_expected  # (n_m, budget)

            # Find argmax over c dimension
            best_idx = np.argmax(total_util, axis=1)  # (n_m,)
            best_c = c_mat[np.arange(n_m), best_idx]  # (n_m,)
            best_val = total_util[np.arange(n_m), best_idx]  # (n_m,)

            new_V[z] = best_val
            new_c[z] = best_c

        consumption[t] = new_c
        V = new_V

    # Final period must consume all: c[5] = m_grid
    consumption[horizon - 1] = m_grid[None, :]

    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c_row = consumption[t, z]
            # c must be > 0 and <= m
            c_row = np.maximum(c_row, 1e-12)
            c_row = np.minimum(c_row, m_grid + 1e-10)
            consumption[t, z] = c_row

    # Ensure finite
    consumption = np.where(np.isfinite(consumption), consumption, 1e-6)

    return consumption
# EVOLVE-BLOCK-END
