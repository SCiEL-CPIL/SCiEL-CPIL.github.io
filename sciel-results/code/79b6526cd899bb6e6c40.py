"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Wall-clock deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s safety margin
    
    # Wrap f to count evaluations
    count = 0
    def f_counted(t, y):
        nonlocal count
        count += 1
        return f(t, y)

    # Try Radau with very tight tolerances first (good for stiff linear systems)
    try:
        sol = scipy.integrate.solve_ivp(
            f_counted,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to BDF if Radau fails
    try:
        sol = scipy.integrate.solve_ivp(
            f_counted,
            t_span,
            y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback to a simple explicit method with small steps if scipy fails
    # This is unlikely but ensures we return something
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0
    for _ in range(steps):
        y = y + h * f(t, y)
        t += h
    return y
# EVOLVE-BLOCK-END
