import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] given initial condition y0.
    Uses high-order adaptive solvers (DOP853 and Radau) with tight tolerances.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span

    y_dop853 = None
    y_radau = None
    
    # Try DOP853 (Explicit, high order)
    try:
        sol1 = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-10, atol=1e-12)
        if sol1.success:
            y_dop853 = sol1.y[:, -1]
    except Exception:
        pass

    if time.time() > deadline - 10.0:
        if y_dop853 is not None:
            return y_dop853
        return _fallback_solve(f, t_span, y0, start_time, deadline)

    # Try Radau (Implicit, for stiff problems)
    try:
        sol2 = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        if sol2.success:
            y_radau = sol2.y[:, -1]
    except Exception:
        pass

    # If both succeeded, check agreement
    if y_dop853 is not None and y_radau is not None:
        diff = np.linalg.norm(y_dop853 - y_radau)
        scale = 1.0 + np.linalg.norm(y_dop853)
        if diff < 1e-8 * scale:
            # They agree, return the higher order one (DOP853 is order 8, Radau is order 5)
            return y_dop853
        else:
            # Disagreement: run a tighter solve to resolve
            try:
                sol3 = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
                if sol3.success:
                    return sol3.y[:, -1]
            except Exception:
                pass
            # If tight solve fails, return DOP853
            return y_dop853

    if y_dop853 is not None:
        return y_dop853
    if y_radau is not None:
        return y_radau

    return _fallback_solve(f, t_span, y0, start_time, deadline)

def _fallback_solve(f, t_span, y0, start_time, deadline):
    """
    Fallback solver using RK4 if adaptive solvers fail.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
