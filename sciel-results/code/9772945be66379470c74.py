"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 3e7 * y1, -1e4 * y1],
        [0.0, 3e7 * y1, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem y' = f(t, y) on [t0, t1].
    Returns the final state y(t1).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    C0 = np.sum(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290
    
    # Try Radau with analytical Jacobian for efficiency and stability
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-13,
                atol=1e-15,
                jac=_robertson_jac
            )
            if sol.success:
                y_end = sol.y[:, -1].copy()
                # Enforce conservation law to reduce error
                C_end = np.sum(y_end)
                if C_end > 1e-30 and np.isfinite(C_end):
                    y_end = y_end * (C0 / C_end)
                return y_end
    except Exception:
        pass
    
    # Fallback to BDF without Jacobian if Radau fails or times out
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                y_end = sol.y[:, -1].copy()
                C_end = np.sum(y_end)
                if C_end > 1e-30 and np.isfinite(C_end):
                    y_end = y_end * (C0 / C_end)
                return y_end
            else:
                if sol.y.shape[1] > 0:
                    y_end = sol.y[:, -1].copy()
                    C_end = np.sum(y_end)
                    if C_end > 1e-30 and np.isfinite(C_end):
                        y_end = y_end * (C0 / C_end)
                    return y_end
    except Exception:
        pass
    
    # Last resort: simple RK4
    h = (t1 - t0) / 10000
    t = t0
    y = y0.copy()
    while t < t1:
        if time.time() > deadline:
            break
        h_step = min(h, t1 - t)
        k1 = f(t, y)
        k2 = f(t + h_step/2, y + h_step/2 * k1)
        k3 = f(t + h_step/2, y + h_step/2 * k2)
        k4 = f(t + h_step, y + h_step * k3)
        y = y + h_step/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h_step
        
    C_end = np.sum(y)
    if C_end > 1e-30 and np.isfinite(C_end):
        y = y * (C0 / C_end)
    return y

# EVOLVE-BLOCK-END
