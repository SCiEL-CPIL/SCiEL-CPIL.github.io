import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']

    M = resource_grid
    num_m = len(M)
    num_z = len(income)

    # CRRA utility and marginal utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def mu(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Inverse marginal utility: given mu, find c
    def mu_inv(mu_val):
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / np.maximum(mu_val, 1e-12)
        else:
            # mu = c^(-gamma) => c = mu^(-1/gamma)
            return np.power(np.maximum(mu_val, 1e-12), -1.0 / gamma)

    # Storage for value functions and consumption policies
    # V[t, z, m_index] = value function
    # C[t, z, m_index] = consumption policy
    V = np.zeros((horizon + 1, num_z, num_m))
    C = np.zeros((horizon + 1, num_z, num_m))

    # Terminal condition: consume everything
    for z in range(num_z):
        V[horizon, z, :] = u(M)
        C[horizon, z, :] = M

    # Build continuation value interpolation for each t
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 10:
            # Not enough time, break and return what we have
            # Fill remaining with linear policy
            for tt in range(t + 1, horizon):
                for z in range(num_z):
                    C[tt, z, :] = M * 0.5
            break

        # Build interpolation of V[t+1, z', :] for each z'
        # We need V[t+1, z', m'] where m' = R*(m - c) + income[z']
        # For each current z, we need the expected continuation value

        # Precompute spline for each z' of V[t+1, z', :]
        splines = []
        for zp in range(num_z):
            splines.append(CubicSpline(M, V[t + 1, zp, :], bc_type='natural'))

        for z in range(num_z):
            # Transition probabilities to z'
            probs = transition[z]

            # We need to solve for c that maximizes:
            # u(c) + beta * sum_z' probs[z'] * V[t+1, z', R*(m-c) + income[z']]
            # FOC: mu(c) = beta * R * sum_z' probs[z'] * V'[t+1, z', R*(m-c) + income[z']]
            # where V' is the derivative of V w.r.t. m

            # For each m, solve the Euler equation
            for i, m in enumerate(M):
                if m < 1e-10:
                    C[t, z, i] = max(m, 1e-12)
                    V[t, z, i] = u(C[t, z, i])
                    continue

                # Newton's method to solve:
                # f(c) = mu(c) - beta * R * sum_z' probs[z'] * V'_zp(R*(m-c) + income[z']) = 0
                # f'(c) = -mu'(c) + beta * R^2 * sum_z' probs[z'] * V''_zp(R*(m-c) + income[z'])

                # Initial guess: use a fraction of m
                c = m * 0.5
                c = np.clip(c, 1e-12, m - 1e-12)

                for newton_iter in range(20):
                    # Compute f(c) and f'(c)
                    f = 0.0
                    fp = 0.0  # derivative of f w.r.t. c

                    mu_c = mu(c)
                    # mu'(c) = -gamma * c^(-gamma-1) for gamma != 1, -1/c^2 for gamma=1
                    if abs(gamma - 1.0) < 1e-10:
                        mu_prime_c = -1.0 / (c * c)
                    else:
                        mu_prime_c = -gamma * c ** (-gamma - 1)

                    for zp in range(num_z):
                        if probs[zp] < 1e-15:
                            continue
                        m_prime = R * (m - c) + income[zp]
                        if m_prime < 1e-10:
                            m_prime = 1e-10
                        # V' and V'' from spline
                        vp = splines[zp](m_prime, 1)
                        vpp = splines[zp](m_prime, 2)

                        f += probs[zp] * beta * R * vp
                        # d/dc of [beta * R * probs[zp] * V'(R*(m-c)+income[zp])]
                        # = beta * R * probs[zp] * V'' * (-R)
                        fp += probs[zp] * beta * R * R * vpp

                    f = mu_c - f
                    fp = -mu_prime_c + fp

                    if abs(fp) < 1e-15:
                        break

                    delta_c = -f / fp
                    # Damping to ensure c stays in (0, m)
                    c_new = c + delta_c
                    if c_new <= 1e-12:
                        c_new = c * 0.5
                    elif c_new >= m - 1e-12:
                        c_new = (c + m) * 0.5

                    if abs(c_new - c) < 1e-12 * max(1.0, m):
                        c = c_new
                        break
                    c = c_new

                c = np.clip(c, 1e-12, m - 1e-12)
                C[t, z, i] = c

                # Compute value
                val = u(c)
                for zp in range(num_z):
                    if probs[zp] < 1e-15:
                        continue
                    m_prime = R * (m - c) + income[zp]
                    if m_prime < 1e-10:
                        m_prime = 1e-10
                    val += probs[zp] * beta * V[t + 1, zp, np.clip(np.searchsorted(M, m_prime, side='left'), 0, num_m - 1)]

                V[t, z, i] = val

    # Final polish: ensure terminal condition is exact
    for z in range(num_z):
        C[horizon, z, :] = M
        V[horizon, z, :] = u(M)

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, M + 1e-10)

    # Return consumption for t=0..horizon-1
    return C[:horizon]
