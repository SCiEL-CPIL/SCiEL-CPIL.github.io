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
    fine_M = 2000
    fine_grid = np.geomspace(0.001, 40, fine_M)

    # Consumption fractions for initial search
    c_frac = np.linspace(0.001, 1.0, 300)

    # Precompute utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def utility_deriv(c):
            return 1.0 / np.maximum(c, 1e-12)
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.maximum(c, 1e-12) ** one_minus_gamma / one_minus_gamma
        def utility_deriv(c):
            return np.maximum(c, 1e-12) ** (-gamma)

    # Value function and consumption on fine grid
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                # Terminal: consume all
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            # For this (t, z), compute optimal c for each m
            m_vals = fine_grid[:, None]  # (fine_M, 1)
            c_vals = m_vals * c_frac[None, :]  # (fine_M, len(c_frac))

            # Utility of c
            u_c = utility(c_vals)  # (fine_M, len(c_frac))

            # m' = R*(m - c) + income[z']
            m_minus_c = m_vals - c_vals  # (fine_M, len(c_frac))
            m_prime = R * m_minus_c[:, :, None] + income[None, None, :]  # (fine_M, len(c_frac), Z)
            m_prime = np.maximum(m_prime, 1e-12)
            m_prime_clipped = np.clip(m_prime, fine_grid[0], fine_grid[-1])

            # Interpolate V[t+1, z', :] for each z'
            V_next = V[t + 1, :, :]  # (Z, fine_M)
            cont = np.zeros((fine_M, len(c_frac)))

            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                mp_flat = m_prime_clipped[:, :, zp].ravel()
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
            if time.time() > deadline - 10.0:
                break

        # Check deadline
        if time.time() > deadline - 10.0:
            break

    # Newton refinement pass
    # For each t, z, refine consumption using Euler equation
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 15.0:
            break
        for z in range(Z):
            if t == horizon - 1:
                continue

            V_next = V[t + 1, :, :]  # (Z, fine_M)
            c_refined = consumption_fine[t, z, :].copy()  # (fine_M,)

            # Newton iterations
            for newton_iter in range(5):
                if time.time() > deadline - 15.0:
                    break

                c_curr = c_refined  # (fine_M,)
                m_vals = fine_grid  # (fine_M,)

                # Compute m' for each z'
                m_prime = R * (m_vals[:, None] - c_curr[:, None]) + income[None, :]  # (fine_M, Z)
                m_prime = np.maximum(m_prime, 1e-12)
                m_prime_clipped = np.clip(m_prime, fine_grid[0], fine_grid[-1])

                # Interpolate V_next and its derivative
                # V_next_deriv: dV/dm for each z'
                # We need to compute dV/dm numerically or analytically
                # For now, use numerical differentiation
                V_next_deriv = np.zeros((Z, fine_M))
                for zp in range(Z):
                    # Central difference for interior, one-sided for boundaries
                    dV = np.zeros(fine_M)
                    # Use np.gradient for simplicity
                    dV = np.gradient(V_next[zp, :], fine_grid, edge_order=2)
                    V_next_deriv[zp, :] = dV

                # Compute gradient of objective w.r.t. c
                # d/dc [u(c) + beta * sum_z' p_z' V(t+1, z', R*(m-c)+income[z'])]
                # = u'(c) - beta * R * sum_z' p_z' V'(t+1, z', m')

                u_deriv = utility_deriv(c_curr)  # (fine_M,)

                # Compute sum over z' of p_z' * V'(m')
                grad_cont = np.zeros(fine_M)
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    mp_flat = m_prime_clipped[:, zp]
                    v_deriv_flat = np.interp(mp_flat, fine_grid, V_next_deriv[zp, :])
                    grad_cont += p * v_deriv_flat

                grad = u_deriv - beta * R * grad_cont  # (fine_M,)

                # Compute Hessian (second derivative) for Newton step
                # d2/dc2 [u(c) + beta * sum_z' p_z' V(t+1, z', R*(m-c)+income[z'])]
                # = u''(c) + beta * R^2 * sum_z' p_z' V''(t+1, z', m')

                if abs(gamma - 1.0) < 1e-10:
                    u_second = -1.0 / np.maximum(c_curr, 1e-12) ** 2
                else:
                    u_second = -gamma * np.maximum(c_curr, 1e-12) ** (-gamma - 1)

                # Numerical second derivative of V
                V_next_second = np.zeros((Z, fine_M))
                for zp in range(Z):
                    # Use second-order central difference
                    d2V = np.zeros(fine_M)
                    # Simple approach: use gradient of gradient
                    dV = np.gradient(V_next[zp, :], fine_grid, edge_order=2)
                    d2V = np.gradient(dV, fine_grid, edge_order=2)
                    V_next_second[zp, :] = d2V

                hess_cont = np.zeros(fine_M)
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    mp_flat = m_prime_clipped[:, zp]
                    v_second_flat = np.interp(mp_flat, fine_grid, V_next_second[zp, :])
                    hess_cont += p * v_second_flat

                hess = u_second + beta * R ** 2 * hess_cont  # (fine_M,)

                # Newton step: c_new = c - grad / hess
                # Be careful with sign and magnitude
                valid = np.abs(hess) > 1e-10
                step = np.zeros(fine_M)
                step[valid] = grad[valid] / hess[valid]

                # Limit step size
                max_step = 0.1 * c_curr
                step = np.clip(step, -max_step, max_step)

                c_new = c_curr - step
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m_vals - 1e-12)  # Ensure m - c > 0

                c_refined = c_new

            consumption_fine[t, z, :] = c_refined

            # Update V with refined consumption
            c_vals_ref = c_refined[:, None]
            m_vals = fine_grid[:, None]
            u_c_ref = utility(c_vals_ref)
            m_minus_c_ref = m_vals - c_vals_ref
            m_prime_ref = R * m_minus_c_ref + income[None, :]
            m_prime_ref = np.maximum(m_prime_ref, 1e-12)
            m_prime_ref_clipped = np.clip(m_prime_ref, fine_grid[0], fine_grid[-1])

            cont_ref = np.zeros(fine_M)
            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                mp_flat = m_prime_ref_clipped[:, zp]
                v_flat = np.interp(mp_flat, fine_grid, V_next[zp, :])
                cont_ref += p * v_flat

            V[t, z, :] = u_c_ref[:, 0] + beta * cont_ref

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
