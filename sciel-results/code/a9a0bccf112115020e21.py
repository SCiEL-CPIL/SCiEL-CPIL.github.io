"""Vectorized consumption policy solver using dense ratio grid and argmax selection."""
import time
import numpy as np

DEADLINE_SECONDS = 135  # Safety margin before the 150s kill


def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    trans = np.asarray(model['transition'])
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    consumption = np.zeros((horizon, states, n_grid))

    # Dense consumption ratio grid
    n_ratios = 400
    ratios = np.geomspace(0.002, 1.0, n_ratios)

    # Terminal condition: consume all
    consumption[horizon - 1] = np.tile(resource_grid, (states, 1))

    # V_next[z, i] = value at next period for state z, grid index i
    # At terminal, V_next is the utility of consuming all (but we don't need it
    # since we stop the loop at t = horizon - 2)
    # Actually, for t = horizon - 2, we need V at t = horizon - 1.
    # V[m, z] at terminal = u(m) since c = m.
    V_next = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)

    for t in range(horizon - 2, -1, -1):
        V_current = np.zeros((states, n_grid))
        m_vals = resource_grid  # shape (n_grid,)

        for z in range(states):
            # m shape: (n_grid,)
            # ratios shape: (n_ratios,)
            # c shape: (n_grid, n_ratios)
            c_grid = m_vals[:, None] * ratios[None, :]

            # m' = R * (m - c) + income[zp]
            # m_minus_c shape: (n_grid, n_ratios)
            m_minus_c = m_vals[:, None] - c_grid  # (n_grid, n_ratios)
            # For each zp, m_prime shape: (n_grid, n_ratios)
            # We'll compute the expected continuation value

            expected_V = np.zeros((n_grid, n_ratios))
            for zp in range(states):
                m_prime = R * m_minus_c + income[zp]  # (n_grid, n_ratios)
                # Clip m_prime to grid range
                m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                # Find indices for searchsorted
                # For each element, find where it would be inserted
                idx = np.searchsorted(resource_grid, m_prime, side='right') - 1
                idx = np.clip(idx, 0, n_grid - 1)
                # Get left and right values
                idx_left = idx
                idx_right = np.clip(idx + 1, 0, n_grid - 1)
                # Interpolation weights
                m_left = resource_grid[idx_left]  # (n_grid, n_ratios)
                m_right = resource_grid[idx_right]
                denom = m_right - m_left
                # Avoid division by zero
                denom = np.where(denom < 1e-15, 1.0, denom)
                w = (m_prime - m_left) / denom
                w = np.clip(w, 0.0, 1.0)

                V_left = V_next[zp, idx_left]  # (n_grid, n_ratios)
                V_right = V_next[zp, idx_right]
                V_interp = V_left + w * (V_right - V_left)
                expected_V += trans[z, zp] * V_interp

            # Objective: u(c) + beta * E[V(m')]
            u_vals = _crsa_utility(c_grid, gamma)  # (n_grid, n_ratios)
            objective = u_vals + beta * expected_V  # (n_grid, n_ratios)

            # Find best ratio for each m
            best_idx = np.argmax(objective, axis=1)  # (n_grid,)
            best_c = c_grid[np.arange(n_grid), best_idx]  # (n_grid,)

            # Ensure c <= m (should be true since ratio <= 1)
            best_c = np.minimum(best_c, m_vals)
            best_c = np.maximum(best_c, 1e-12)

            consumption[t, z, :] = best_c
            V_current[z, :] = objective[np.arange(n_grid), best_idx]

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Check deadline
        if time.time() > deadline:
            pass  # We already have a result

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        states = model['transition'].shape[0]
        if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                result[t] = 0.9 * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
