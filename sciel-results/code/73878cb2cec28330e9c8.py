"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using an implicit
    Radau method with tight tolerances. Falls back to BDF if Radau fails.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    deadline = time.time() + 280  # safety margin before 300s limit
    
    # Try Radau with very tight tolerances for high accuracy
    try:
        sol = scipy.integrate.solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            max_step=np.inf
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau failed or timed out, try BDF with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f, (t0, t1), y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12,
            max_step=np.inf
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: Radau with moderate tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-8,
            atol=1e-10,
            max_step=np.inf
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Absolute fallback: simple RK4 with fixed steps
    h = (t1 - t0) / 1000
    t, y = t0, y0.copy()
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if time.time() > deadline:
            break
    return y
# EVOLVE-BLOCK-END
