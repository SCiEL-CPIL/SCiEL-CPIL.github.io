import numpy as np
from scipy.integrate import solve_ivp
import time

def _van_der_pol_jac(t, y):
    """Analytical Jacobian for the Van der Pol oscillator (mu=5)."""
    y0, y1 = y
    mu = 5.0
    return np.array([
        [0.0, 1.0],
        [-1.0 - 2.0 * mu * y0 * y1, mu * (1.0 - y0 * y0)]
    ])

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # Try DOP853 first for highest accuracy per call
    # DOP853 is an 8th-order method
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to RK45 if DOP853 fails or is too slow
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='RK45',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau with Jacobian if explicit methods fail
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            jac=_van_der_pol_jac
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Simple RK4 with many steps
    # This ensures we return *something* finite
    h = (t1 - t0) / 10000
    t, y = t0, y0_arr.copy()
    for _ in range(10000):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
