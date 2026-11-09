"""Bootstrap: direct Robertson ODE solver using high-accuracy SciPy Radau."""
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # Robertson is a classic stiff problem. Radau (implicit Runge-Kutta, L-stable)
    # is excellent for stiff IVPs and typically uses far fewer f-calls than
    # explicit methods while achieving high accuracy.
    #
    # Strategy: use Radau with very tight tolerances to minimize relative error.
    # If it fails or times out, fall back to BDF, then LSODA, then RK45.
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        solve_ivp = None

    if solve_ivp is not None:
        # Primary: Radau with tight tolerances
        try:
            sol = solve_ivp(
                f, [t0, t1], y0,
                method='Radau',
                rtol=1e-13,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass

        # Secondary: BDF with tight tolerances
        try:
            sol = solve_ivp(
                f, [t0, t1], y0,
                method='BDF',
                rtol=1e-13,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass

        # Tertiary: LSODA
        try:
            sol = solve_ivp(
                f, [t0, t1], y0,
                method='LSODA',
                rtol=1e-13,
                atol=1e-15
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass

    # Final fallback: manual RK45 with moderate tolerances
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 1000.0
    rtol = 1e-10
    atol = 1e-12

    while t < t1 and time.time() < deadline:
        h = min(h, t1 - t)
        if h <= 0:
            break

        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        k5 = f(t + h / 5, y + h / 5 * k4)
        k6 = f(t + 3 * h / 5, y + 3 * h / 5 * (k1 + 4.0 / 3.0 * (k4 - k1)))

        y_new = y + h * (k1 + 3 * k2 + 3 * k3 + k4) / 8.0
        err = h * (k4 - k3) / 32.0
        err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))

        if err_norm <= max(rtol, atol):
            y = y_new
            t += h
            # Grow step if error is small
            if err_norm > 0:
                h *= min(4.0, 0.9 * (rtol / err_norm) ** 0.2)
            else:
                h *= 4.0
        else:
            # Shrink step
            h *= max(0.1, 0.9 * (rtol / max(err_norm, 1e-15)) ** 0.2)

    return y
