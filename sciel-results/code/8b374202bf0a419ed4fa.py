import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0

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

    # Dense internal grid for value function
    n_int = 1000
    int_grid = np.geomspace(0.001, 40, n_int)

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # V_int[t][z] : value function on internal grid for state z
    # P_out[t][z] : policy on the output (resource) grid
    V_int = [None] * T
    P_out = [None] * T

    # Terminal period: consume everything
    V_int[T - 1] = utility(int_grid)[np.newaxis, :].repeat(2, axis=0)
    P_out[T - 1] = grid[np.newaxis, :].repeat(2, axis=0)

    # Number of consumption candidates for maximization
    n_cand = 200

    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 3:
            # Fallback: linear interpolation from terminal
            for t_rem in range(t, -1, -1):
                P_out[t_rem] = P_out[T - 1].copy()
                V_int[t_rem] = V_int[T - 1].copy()
            break

        V_next = V_int[t + 1]  # shape (2, n_int)
        V_t_int = np.zeros((n_z, n_int))
        P_t_out = np.zeros((n_z, n_grid))

        for z in range(n_z):
            # --- Compute policy on output grid ---
            m_out = grid  # shape (n_grid,)
            # Consumption candidates: fraction of m
            frac = np.linspace(0.001, 1.0, n_cand)  # shape (n_cand,)
            c_cands = frac[:, np.newaxis] * m_out[np.newaxis, :]  # (n_cand, n_grid)
            c_cands = np.clip(c_cands, c_min, m_out[np.newaxis, :])

            # For each candidate c, compute next_m for each z'
            # next_m[zp, i, j] = R*(m_out[j] - c_cands[i,j]) + income[zp]
            # Shape: (n_z, n_cand, n_grid)
            next_m = R * (m_out[np.newaxis, np.newaxis, :] - c_cands[np.newaxis, :, :]) + income[:, np.newaxis, np.newaxis]

            # Interpolate V_next on internal grid
            # V_interp[zp, i, j] = V_next[zp, next_m[zp,i,j]]
            V_interp = np.array([np.interp(next_m[zp].ravel(), int_grid, V_next[zp]).reshape(n_cand, n_grid) for zp in range(n_z)])
            # Shape: (n_z, n_cand, n_grid)

            # Continuation value: sum over z' of transition[z, zp] * V_interp[zp]
            cont = np.sum(transition[z, :][:, np.newaxis, np.newaxis] * V_interp, axis=0)
            # Shape: (n_cand, n_grid)

            # Total objective: u(c) + beta * cont
            obj = utility(c_cands) + beta * cont  # (n_cand, n_grid)

            # Argmax over candidates for each m
            best_idx = np.argmax(obj, axis=0)  # (n_grid,)
            c_opt_out = c_cands[best_idx, np.arange(n_grid)]  # (n_grid,)
            c_opt_out = np.clip(c_opt_out, c_min, m_out)

            P_t_out[z, :] = c_opt_out

            # --- Compute value function on internal grid ---
            m_int = int_grid  # shape (n_int,)
            c_cands_int = frac[:, np.newaxis] * m_int[np.newaxis, :]  # (n_cand, n_int)
            c_cands_int = np.clip(c_cands_int, c_min, m_int[np.newaxis, :])

            next_m_int = R * (m_int[np.newaxis, np.newaxis, :] - c_cands_int[np.newaxis, :, :]) + income[:, np.newaxis, np.newaxis]
            V_interp_int = np.array([np.interp(next_m_int[zp].ravel(), int_grid, V_next[zp]).reshape(n_cand, n_int) for zp in range(n_z)])
            cont_int = np.sum(transition[z, :][:, np.newaxis, np.newaxis] * V_interp_int, axis=0)
            obj_int = utility(c_cands_int) + beta * cont_int  # (n_cand, n_int)

            best_idx_int = np.argmax(obj_int, axis=0)
            c_opt_int = c_cands_int[best_idx_int, np.arange(n_int)]

            V_t_int[z, :] = obj_int[best_idx_int, np.arange(n_int)]

        V_int[t] = V_t_int
        P_out[t] = P_t_out

    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P_out[t]

    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)

    # Final period: must consume everything
    result[T - 1, :, :] = grid[np.newaxis, :]

    return result
