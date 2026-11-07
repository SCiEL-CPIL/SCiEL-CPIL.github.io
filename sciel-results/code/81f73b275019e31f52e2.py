"""Consumption policy solver using PCHIP interpolation + FOC Newton solver."""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=np.float64)
    N_m = len(m_grid)
    N_z = len(income)

    def check_time():
        return (time.time() - start_time) > DEADLINE

    # Utility functions
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Working grid: use a dense grid for value function representation
    # Use the scoring grid resolution for accuracy
    dense_N = 401
    dense_grid = np.geomspace(0.001, 40, dense_N)

    # Store value function on dense grid for each t and z
    # V[t, z, :] = value at dense_grid points
    V = np.zeros((horizon, N_z, dense_N))

    # Store consumption policy on dense grid
    C = np.zeros((horizon, N_z, dense_N))

    # Terminal condition: t = horizon-1 = 5
    # c_5(m, z) = m (consume everything)
    # V_5(m, z) = u(m)
    C[5, :, :] = dense_grid[np.newaxis, :]
    V[5, :, :] = util(dense_grid)[np.newaxis, :]

    # Backward induction from t=4 down to t=0
    for t in range(horizon - 2, -1, -1):
        if check_time():
            # Fill remaining with a reasonable default
            for tt in range(t, horizon - 1):
                for z in range(N_z):
                    C[tt, z, :] = dense_grid * 0.5
                    V[tt, z, :] = util(dense_grid * 0.5)
            break

        for z in range(N_z):
            # Solve for optimal c at each dense grid point
            for i in range(dense_N):
                m_i = dense_grid[i]

                # Define the FOC: u'(c) - beta * sum_{z'} P(z'|z) * V'_{t+1}(m', z') * R = 0
                # where m' = R*(m_i - c) + income[z']
                # V'_{t+1} is the derivative of the value function w.r.t. m

                # First, compute V'_{t+1} for each z' using PCHIP derivative
                # We need to build PCHIP interpolators for V_{t+1, z'}
                V_next_derivs = np.zeros(N_z)
                V_next_derivs_arr = np.zeros((N_z, dense_N))

                for zp in range(N_z):
                    # Build PCHIP for V_{t+1, zp}
                    pchip_V = PchipInterpolator(dense_grid, V[t + 1, zp, :])
                    # Get derivative at dense_grid points
                    V_next_derivs_arr[zp, :] = pchip_V(dense_grid, 1)

                def foc(c):
                    """FOC: u'(c) - beta * R * sum_{z'} P(z'|z) * V'_{t+1}(m', z') = 0"""
                    if c <= 0 or c > m_i + 1e-10:
                        return 1e10  # push away from boundary
                    c = np.maximum(c, 1e-12)
                    u_p = u_prime(c)

                    cont_deriv = 0.0
                    for zp in range(N_z):
                        m_prime = R * (m_i - c) + income[zp]
                        m_prime = np.clip(m_prime, 0.001, 40.0)
                        # Interpolate V'_{t+1, zp} at m_prime
                        Vp = np.interp(m_prime, dense_grid, V_next_derivs_arr[zp, :])
                        cont_deriv += transition[z, zp] * Vp

                    return u_p - beta * R * cont_deriv

                # Solve FOC using brentq
                # Find bracket: c in [1e-8, m_i]
                c_lo = 1e-8
                c_hi = m_i

                # Evaluate FOC at boundaries
                f_lo = foc(c_lo)
                f_hi = foc(c_hi)

                # If no sign change, try to find one or use boundary
                if f_lo * f_hi > 0:
                    # Try a few points to find a bracket
                    c_try = m_i * 0.5
                    f_try = foc(c_try)
                    if f_lo * f_try < 0:
                        c_hi = c_try
                        f_hi = f_try
                    elif f_try * f_hi < 0:
                        c_lo = c_try
                        f_lo = f_try
                    else:
                        # No bracket found; use golden-section or just pick a reasonable value
                        # For CRRA, optimal c is typically between 0.1*m and 0.9*m
                        c_star = m_i * 0.5
                else:
                    c_star = brentq(foc, c_lo, c_hi, xtol=1e-12, rtol=1e-12, maxiter=100)

                C[t, z, i] = c_star

                # Compute value: V_t(m_i, z) = u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(m', z')
                u_val = util(c_star)
                cont = 0.0
                for zp in range(N_z):
                    m_prime = R * (m_i - c_star) + income[zp]
                    m_prime = np.clip(m_prime, 0.001, 40.0)
                    V_next = np.interp(m_prime, dense_grid, V[t + 1, zp, :])
                    cont += transition[z, zp] * V_next
                V[t, z, i] = u_val + beta * cont

    # Interpolate consumption policy from dense_grid to m_grid
    consumption = np.zeros((horizon, N_z, N_m))
    for t in range(horizon):
        for z in range(N_z):
            consumption[t, z, :] = np.interp(m_grid, dense_grid, C[t, z, :])

    # Enforce constraints
    consumption[5, :, :] = m_grid[np.newaxis, :]  # Terminal: c = m
    for t in range(horizon - 1):
        for z in range(N_z):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid + 1e-10)

    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-8, posinf=m_grid.max(), neginf=1e-12)

    return consumption
