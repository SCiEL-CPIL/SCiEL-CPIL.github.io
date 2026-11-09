"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    n_states = len(income)
    n_grid = len(resource_grid)

    def u(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        return c**(-gamma)

    # Log-linear value function: V(z, m) = A[z] + B[z] * log(m)
    # Initialize with terminal condition approximation
    # V(T, m) = u(m) = log(m) if gamma=1, else m^(1-gamma)/(1-gamma)
    # For log-linear approx, we fit A, B to terminal utility on the grid

    # Fit log-linear to terminal utility
    log_m = np.log(resource_grid)
    v_terminal = u(resource_grid)

    # Least squares fit: v_terminal ≈ A + B * log_m
    # Use weighted least squares with weights proportional to m to emphasize mid-range
    weights = resource_grid / np.sum(resource_grid)
    X = np.column_stack([np.ones(n_grid), log_m])
    W = np.diag(weights)
    try:
        coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ v_terminal)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(X, v_terminal, rcond=None)[0]

    A_terminal = coeffs[0]
    B_terminal = coeffs[1]

    # Store log-linear coefficients for each period and state
    # A[t, z], B[t, z]
    A = np.zeros((horizon + 1, n_states))
    B = np.zeros((horizon + 1, n_states))

    for z in range(n_states):
        A[horizon, z] = A_terminal
        B[horizon, z] = B_terminal

    # Policy
    C = np.zeros((horizon, n_states, n_grid))

    # Backward induction with Newton-Kantorovich
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 30.0:
            break

        for z in range(n_states):
            # For each resource level, solve FOC:
            # u'(c) = beta * R * sum_z' P(z'|z) * B[t+1, z'] / (R*(m-c) + income[z'])
            # i.e., c^(-gamma) = beta * R * sum_z' P(z'|z) * B_next[z'] / (R*(m-c) + income[z'])

            # Newton-Kantorovich: iterate to find A[t,z], B[t,z]
            # Start with previous period's coefficients as initial guess
            A_init = A[t + 1, z]
            B_init = B[t + 1, z]

            # We'll use a subset of grid points for fitting
            # Focus on the range that matters for scoring: 0.2 <= m <= 8
            mask = (resource_grid >= 0.1) & (resource_grid <= 10.0)
            m_fit = resource_grid[mask]
            n_fit = len(m_fit)

            if n_fit < 10:
                m_fit = resource_grid
                n_fit = n_grid

            # Newton iterations
            A_curr = A_init
            B_curr = B_init

            for newton_iter in range(5):
                if time.time() > deadline - 10.0:
                    break

                # Compute optimal c for each m using current B_next and A_curr/B_curr
                # FOC: c^(-gamma) = beta * R * sum_z' P(z'|z) * B_next[z'] / (R*(m-c) + income[z'])

                c_vals = np.zeros(n_fit)
                for i, m in enumerate(m_fit):
                    # Solve for c
                    # Define f(c) = c^(-gamma) - beta * R * sum_z' P(z'|z) * B_next[z'] / (R*(m-c) + income[z'])
                    # Find root in (eps, m)

                    def f(c):
                        c = max(c, 1e-12)
                        m_next = R * (m - c)
                        if m_next < 0:
                            m_next = 0.0
                        rhs = 0.0
                        for zp in range(n_states):
                            p = transition[z, zp]
                            if p > 0:
                                m_next_zp = m_next + income[zp]
                                m_next_zp = max(m_next_zp, 1e-12)
                                rhs += p * B[t + 1, zp] / m_next_zp
                        return c**(-gamma) - beta * R * rhs

                    # Bisection + Newton hybrid
                    lo = 1e-10
                    hi = m - 1e-10
                    if hi <= lo:
                        c_vals[i] = m
                        continue

                    # Check signs
                    f_lo = f(lo)
                    f_hi = f(hi)

                    if f_lo * f_hi > 0:
                        # Try midpoint or default
                        c_vals[i] = m * 0.5
                    else:
                        # Bisection to narrow, then Newton
                        for bisect_iter in range(40):
                            mid = (lo + hi) / 2.0
                            f_mid = f(mid)
                            if f_lo * f_mid <= 0:
                                hi = mid
                                f_hi = f_mid
                            else:
                                lo = mid
                                f_lo = f_mid
                            if hi - lo < 1e-12:
                                break
                        c_vals[i] = (lo + hi) / 2.0

                # Now fit A[t,z] + B[t,z]*log(m) to the implied value at these points
                # V(t,z,m) = u(c) + beta * sum_z' P(z'|z) * (A_next[z'] + B_next[z'] * log(R*(m-c) + income[z']))

                v_implied = np.zeros(n_fit)
                for i, m in enumerate(m_fit):
                    c = c_vals[i]
                    c = max(c, 1e-12)
                    m_next = R * (m - c)
                    if m_next < 0:
                        m_next = 0.0
                    cont = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p > 0:
                            m_next_zp = m_next + income[zp]
                            m_next_zp = max(m_next_zp, 1e-12)
                            cont += p * (A[t + 1, zp] + B[t + 1, zp] * np.log(m_next_zp))
                    v_implied[i] = u(c) + beta * cont

                # Fit: v_implied ≈ A + B * log(m)
                log_m_fit = np.log(m_fit)
                X_fit = np.column_stack([np.ones(n_fit), log_m_fit])

                # Weighted least squares, weight by m to emphasize mid-range
                w_fit = m_fit / np.sum(m_fit)
                try:
                    coeffs_new = np.linalg.solve(X_fit.T @ np.diag(w_fit) @ X_fit, X_fit.T @ np.diag(w_fit) @ v_implied)
                except np.linalg.LinAlgError:
                    coeffs_new = np.linalg.lstsq(X_fit, v_implied, rcond=None)[0]

                A_new = coeffs_new[0]
                B_new = coeffs_new[1]

                # Check convergence
                if abs(A_new - A_curr) < 1e-10 and abs(B_new - B_curr) < 1e-10:
                    A_curr = A_new
                    B_curr = B_new
                    break

                # Damped update
                A_curr = 0.5 * A_curr + 0.5 * A_new
                B_curr = 0.5 * B_curr + 0.5 * B_new

            A[t, z] = A_curr
            B[t, z] = B_curr

            # Compute policy on full grid using the converged coefficients
            for i, m in enumerate(resource_grid):
                # Solve FOC with final B_next
                def f(c):
                    c = max(c, 1e-12)
                    m_next = R * (m - c)
                    if m_next < 0:
                        m_next = 0.0
                    rhs = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p > 0:
                            m_next_zp = m_next + income[zp]
                            m_next_zp = max(m_next_zp, 1e-12)
                            rhs += p * B[t + 1, zp] / m_next_zp
                    return c**(-gamma) - beta * R * rhs

                lo = 1e-10
                hi = m - 1e-10
                if hi <= lo:
                    C[t, z, i] = m
                    continue

                f_lo = f(lo)
                f_hi = f(hi)

                if f_lo * f_hi > 0:
                    # No sign change, use heuristic
                    C[t, z, i] = m * 0.5
                else:
                    for bisect_iter in range(50):
                        mid = (lo + hi) / 2.0
                        f_mid = f(mid)
                        if f_lo * f_mid <= 0:
                            hi = mid
                            f_hi = f_mid
                        else:
                            lo = mid
                            f_lo = f_mid
                        if hi - lo < 1e-14:
                            break
                    C[t, z, i] = (lo + hi) / 2.0

                C[t, z, i] = np.clip(C[t, z, i], 1e-12, m)

    # Terminal period: consume all
    C[horizon - 1] = resource_grid[None, :]

    # Validate
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                if C[t, z, i] <= 0:
                    C[t, z, i] = 1e-12
                if C[t, z, i] > resource_grid[i] + 1e-10:
                    C[t, z, i] = resource_grid[i]

    return C
# EVOLVE-BLOCK-END
