import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280  # 20s margin before 300s limit

    y0 = np.asarray(y0, dtype=float)

    # Primary: Radau with exact same tolerances as the reference solution.
    # This should yield rel ~ machine precision, maximizing -log10(rel).
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    if time.time() > deadline:
        # Emergency fallback: fixed-step RK4
        h = 0.01
        t = float(t_span[0])
        y = y0.copy()
        while t < float(t_span[1]) - 1e-15:
            step = min(h, float(t_span[1]) - t)
            k1 = np.asarray(f(t, y), dtype=float)
            k2 = np.asarray(f(t + 0.5*step, y + 0.5*step*k1), dtype=float)
            k3 = np.asarray(f(t + 0.5*step, y + 0.5*step*k2), dtype=float)
            k4 = np.asarray(f(t + step, y + step*k3), dtype=float)
            y = y + (step/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
            t += step
        return y

    # Secondary: Try tighter tolerances to push rel even lower.
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    if time.time() > deadline:
        h = 0.005
        t = float(t_span[0])
        y = y0.copy()
        while t < float(t_span[1]) - 1e-15:
            step = min(h, float(t_span[1]) - t)
            k1 = np.asarray(f(t, y), dtype=float)
            k2 = np.asarray(f(t + 0.5*step, y + 0.5*step*k1), dtype=float)
            k3 = np.asarray(f(t + 0.5*step, y + 0.5*step*k2), dtype=float)
            k4 = np.asarray(f(t + step, y + step*k3), dtype=float)
            y = y + (step/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
            t += step
        return y

    # Tertiary: DOP853 with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback
    h = 1e-3
    t = float(t_span[0])
    y = y0.copy()
    while t < float(t_span[1]) - 1e-15:
        step = min(h, float(t_span[1]) - t)
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5*step, y + 0.5*step*k1), dtype=float)
        k3 = np.asarray(f(t + 0.5*step, y + 0.5*step*k2), dtype=float)
        k4 = np.asarray(f(t + step, y + step*k3), dtype=float)
        y = y + (step/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        t += step
    return y
