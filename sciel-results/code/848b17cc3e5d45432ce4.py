import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # Fine grid for value function
    fine_M = 801
    fine_grid = np.geomspace(0.001, 40, fine_M)

    # c fractions - use a moderate number, we'll refine
    c_frac = np.linspace(0.001, 1.0, 150)

    # Precompute utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.maximum(c, 1e-12) ** one_minus_gamma / one_minus_gamma

    # Value function and consumption on fine grid
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    # Precompute interpolation indices for speed
    # For each fine_grid point m and each c_frac, compute m' for each z'
    # m' = R*(m - c) + income[z']
    # We'll compute this on the fly but vectorized

    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                # Terminal: consume all
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            # For this (t, z), compute optimal c for each m
            # m values: fine_grid
            # c candidates: m * c_frac
            # For each m, we need to evaluate: u(c) + beta * sum_z' P(z'|z) * V[t+1, z', R*(m-c)+income[z']]

            # Vectorize over m and c_frac
            # Shape: (fine_M, len(c_frac))
            m_vals = fine_grid[:, None]  # (fine_M, 1)
            c_vals = m_vals * c_frac[None, :]  # (fine_M, len(c_frac))

            # Filter c > 0 and c <= m
            # c_frac starts at 0.001 so all c > 0
            # c <= m is always true since c_frac <= 1

            # Utility of c
            u_c = utility(c_vals)  # (fine_M, len(c_frac))

            # m' = R*(m - c) + income[z']
            # Shape: (fine_M, len(c_frac), Z)
            m_minus_c = m_vals - c_vals  # (fine_M, len(c_frac))
            m_prime = R * m_minus_c[:, :, None] + income[None, None, :]  # (fine_M, len(c_frac), Z)
            m_prime = np.maximum(m_prime, 1e-12)

            # Clip m_prime to grid range
            m_prime_clipped = np.clip(m_prime, fine_grid[0], fine_grid[-1])

            # Interpolate V[t+1, z', :] for each z'
            # V_next[z', :] has shape (fine_M,)
            # For each (m_idx, c_idx, z'), we need V[t+1, z', m_prime]
            # Vectorized interpolation
            # For each z', interpolate V[t+1, z', :] at m_prime[:, :, z']
            V_next = V[t + 1, :, :]  # (Z, fine_M)
            cont = np.zeros((fine_M, len(c_frac)))
            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                # Interpolate V_next[zp, :] at m_prime[:, :, zp]
                # m_prime[:, :, zp] has shape (fine_M, len(c_frac))
                # We need to handle this efficiently
                # Use np.interp which works on 1D arrays
                # Flatten and unflatten
                mp_flat = m_prime[:, :, zp].ravel()
                v_flat = np.interp(mp_flat, fine_grid, V_next[zp, :])
                v_interp = v_flat.reshape(fine_M, len(c_frac))
                cont += p * v_interp

            # Total value
            total = u_c + beta * cont  # (fine_M, len(c_frac))

            # Find best c for each m
            best_idx = np.argmax(total, axis=1)  # (fine_M,)
            best_c = c_vals[np.arange(fine_M), best_idx]  # (fine_M,)
            V[t, z, :] = total[np.arange(fine_M), best_idx]
            consumption_fine[t, z, :] = best_c

        # Check deadline
        if time.time() > deadline - 30.0:
            break

    # Refine: do a second pass with finer c-grid around the optimal
    # Only if we have time
    if time.time() < deadline - 40.0:
        # Refine with 50 more points around the best
        for t in range(horizon - 2, -1, -1):
            if time.time() > deadline - 40.0:
                break
            for z in range(Z):
                # For each m, refine around the current best
                # Use a local grid
                m_vals = fine_grid
                for m_idx in range(fine_M):
                    m = m_vals[m_idx]
                    c_best = consumption_fine[t, z, m_idx]
                    # Create a fine local grid around c_best
                    # Range: [0.5*c_best, min(1.5*c_best, m)]
                    c_low = max(1e-6, 0.5 * c_best)
                    c_high = min(m, 1.5 * c_best)
                    if c_high - c_low < 1e-6:
                        continue
                    c_local = np.linspace(c_low, c_high, 50)
                    c_local = c_local[(c_local > 0) & (c_local <= m + 1e-10)]
                    if len(c_local) < 2:
                        continue

                    u_c = utility(c_local)
                    m_prime = R * (m - c_local)[:, None] + income[None, :]
                    m_prime = np.maximum(m_prime, 1e-12)
                    m_prime_clipped = np.clip(m_prime, fine_grid[0], fine_grid[-1])

                    cont = np.zeros(len(c_local))
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        v_interp = np.interp(m_prime[:, zp], fine_grid, V[t + 1, zp, :])
                        cont += p * v_interp

                    total = u_c + beta * cont
                    best_idx = np.argmax(total)
                    if total[best_idx] > V[t, z, m_idx]:
                        V[t, z, m_idx] = total[best_idx]
                        consumption_fine[t, z, m_idx] = c_local[best_idx]

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition: c[5] = m
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
