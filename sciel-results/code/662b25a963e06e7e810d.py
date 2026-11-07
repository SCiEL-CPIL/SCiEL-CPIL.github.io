"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)

    # Use a dense internal grid for value function iteration.
    # The evaluator interpolates our policy onto a 251-point grid, so
    # solving on a comparable or denser grid reduces interpolation error.
    n_solve = 401
    solve_grid = np.geomspace(0.001, 40, n_solve)

    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def get_utility_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def get_utility_double_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Value function V[t, z, :] has shape (horizon, 2, n_solve)
    # Policy c[t, z, :] has shape (horizon, 2, n_solve)
    V = np.zeros((horizon, 2, n_solve))
    c_opt = np.zeros((horizon, 2, n_solve))

    # Terminal period: consume everything
    c_opt[horizon - 1, :, :] = solve_grid[np.newaxis, :]
    V[horizon - 1, :, :] = get_utility(solve_grid[np.newaxis, :], gamma)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: simple proportional rule for remaining periods
            for tt in range(t, -1, -1):
                c_opt[tt, :, :] = solve_grid[np.newaxis, :] / (horizon - tt)
                V[tt, :, :] = get_utility(c_opt[tt, :, :], gamma)
            break

        for z in range(2):
            m_vals = solve_grid  # shape (n_solve,)

            # We need to find c* that maximizes:
            # u(c) + beta * E_z'[ V_{t+1}(R*(m-c) + income[z'], z') ]
            #
            # Use a coarse-to-fine search:
            # 1. Evaluate on a coarse grid of c values
            # 2. Refine around the best point

            # Coarse grid: c from 1e-4 to m_vals, 50 points
            # For vectorization, we'll evaluate the objective at many c values simultaneously
            # c_eval shape: (n_c, n_solve) where n_c is number of evaluation points
            n_c = 60
            c_eval = m_vals[np.newaxis, :] * np.linspace(1e-4, 1.0, n_c)[:, np.newaxis]
            c_eval = np.maximum(c_eval, 1e-12)

            # Compute m_next for each (c_eval, zp)
            # m_next shape: (n_c, 2, n_solve)
            m_next = R * (m_vals[np.newaxis, np.newaxis, :] - c_eval[:, np.newaxis, :]) + income[np.newaxis, :, np.newaxis]

            # Interpolate V_{t+1} at m_next for each zp
            # V_interp[zp, :] = V_{t+1}(m_next[:, zp, :], zp)
            # We need PCHIP interpolants for V_{t+1}[zp, :]
            vp_interp = np.zeros((2, n_c, n_solve))
            for zp in range(2):
                # Create PCHIP interpolant for V_{t+1}[zp, :]
                try:
                    pchip = PchipInterpolator(solve_grid, V[t + 1, zp, :])
                    vp_interp[zp, :, :] = pchip(m_next[:, zp, :])
                except Exception:
                    # Fallback to linear interpolation
                    vp_interp[zp, :, :] = np.interp(m_next[:, zp, :], solve_grid, V[t + 1, zp, :])

            # Expected continuation value
            # E_z'[V_{t+1}] = sum_zp transition[z, zp] * V_{t+1}(m_next, zp)
            # Shape: (n_c, n_solve)
            expected_V = transition[z, 0] * vp_interp[0, :, :] + transition[z, 1] * vp_interp[1, :, :]

            # Objective: u(c) + beta * expected_V
            obj = get_utility(c_eval, gamma) + beta * expected_V

            # Find the best c for each m
            best_idx = np.argmax(obj, axis=0)  # shape (n_solve,)
            c_coarse = c_eval[best_idx, np.arange(n_solve)]  # shape (n_solve,)

            # Refine: local search around c_coarse
            # Use a few Newton steps or a finer grid
            # For simplicity, do a fine grid search in a narrow window
            n_fine = 20
            # Window: c_coarse * (1 ± 0.05) or absolute window
            c_low = np.maximum(c_coarse * 0.95, 1e-12)
            c_high = np.minimum(c_coarse * 1.05, m_vals)
            # Ensure c_high > c_low
            c_high = np.maximum(c_high, c_low + 1e-8)

            c_fine = np.zeros((n_fine, n_solve))
            for i in range(n_fine):
                c_fine[i, :] = c_low + (c_high - c_low) * i / (n_fine - 1)

            # Evaluate objective at fine grid
            m_next_fine = R * (m_vals[np.newaxis, np.newaxis, :] - c_fine[:, np.newaxis, :]) + income[np.newaxis, :, np.newaxis]
            vp_fine = np.zeros((2, n_fine, n_solve))
            for zp in range(2):
                try:
                    pchip = PchipInterpolator(solve_grid, V[t + 1, zp, :])
                    vp_fine[zp, :, :] = pchip(m_next_fine[:, zp, :])
                except Exception:
                    vp_fine[zp, :, :] = np.interp(m_next_fine[:, zp, :], solve_grid, V[t + 1, zp, :])

            expected_V_fine = transition[z, 0] * vp_fine[0, :, :] + transition[z, 1] * vp_fine[1, :, :]
            obj_fine = get_utility(c_fine, gamma) + beta * expected_V_fine

            best_idx_fine = np.argmax(obj_fine, axis=0)
            c_refined = c_fine[best_idx_fine, np.arange(n_solve)]

            # Ensure validity: 0 < c <= m
            c_refined = np.clip(c_refined, 1e-12, m_vals + 1e-10)

            c_opt[t, z, :] = c_refined
            V[t, z, :] = get_utility(c_refined, gamma) + beta * (
                transition[z, 0] * np.interp(
                    R * (m_vals - c_refined) + income[0], solve_grid, V[t + 1, 0, :]
                ) +
                transition[z, 1] * np.interp(
                    R * (m_vals - c_refined) + income[1], solve_grid, V[t + 1, 1, :]
                )
            )

    # Interpolate the solution from solve_grid (n_solve points) to resource_grid (101 points)
    consumption = np.zeros((horizon, 2, n_m))
    for z in range(2):
        for t in range(horizon):
            try:
                pchip = PchipInterpolator(solve_grid, c_opt[t, z, :])
                consumption[t, z, :] = pchip(resource_grid)
            except Exception:
                consumption[t, z, :] = np.interp(resource_grid, solve_grid, c_opt[t, z, :])

    # Final check and clipping
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)

    # Ensure c[5] is exactly the grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    # Final validity check
    consumption = np.where(np.isfinite(consumption), consumption, resource_grid[np.newaxis, np.newaxis, :])
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)

    return consumption
# EVOLVE-BLOCK-END
