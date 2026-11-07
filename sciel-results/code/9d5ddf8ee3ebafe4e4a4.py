import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']

    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)

    # Use a coarser internal grid with more consumption fraction points
    n_m_int = 2000
    m_internal = np.geomspace(0.001, 50, n_m_int)

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_m_int))
    # Policy storage: C[t, z, m_idx]
    C = np.zeros((horizon, n_z, n_m_int))

    # Terminal period: consume everything
    V[horizon - 1, :, :] = utility(m_internal)[np.newaxis, :]
    C[horizon - 1, :, :] = m_internal[np.newaxis, :]

    # Grid for consumption fraction s = c/m - use more points for accuracy
    n_s = 4000
    s_grid = np.linspace(0.0001, 1.0, n_s)

    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            m_col = m_internal[:, np.newaxis]  # (n_m_int, 1)
            s_row = s_grid[np.newaxis, :]      # (1, n_s)

            c_mat = m_col * s_row              # (n_m_int, n_s)
            c_mat = np.maximum(c_mat, 1e-12)

            u_c = utility(c_mat)               # (n_m_int, n_s)

            total_val = np.zeros_like(u_c)     # (n_m_int, n_s)

            for zp in range(n_z):
                m_next = R * (m_col - c_mat) + income[zp]  # (n_m_int, n_s)

                # Interpolate V[t+1, zp, :] at m_next
                m_next_flat = m_next.flatten()
                m_next_clipped = np.clip(m_next_flat, m_internal[0], m_internal[-1])

                v_next_flat = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                v_next = v_next_flat.reshape(m_next.shape)

                total_val += beta_trans[z, zp] * v_next

            total_val += u_c

            # Find best s for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(n_m_int), best_idx]
            C[t, z, :] = c_mat[np.arange(n_m_int), best_idx]

            # Enforce c <= m
            C[t, z, :] = np.minimum(C[t, z, :], m_internal)
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)

            # Time check
            if time.time() - start_time > deadline - 5:
                break

        if time.time() - start_time > deadline - 5:
            break

    # Interpolate policy C from m_internal to m_grid_out
    consumption = np.zeros((horizon, n_z, n_m_out))

    for t in range(horizon):
        for z in range(n_z):
            c_out = np.interp(m_grid_out, m_internal, C[t, z, :])
            consumption[t, z, :] = c_out

            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)

    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]

    return consumption
