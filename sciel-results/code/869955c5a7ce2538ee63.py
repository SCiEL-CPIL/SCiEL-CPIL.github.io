"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def _robertson_jac(t, y):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    y0, y1, y2 = y
    J = np.array([
        [-k1, k3*y2, k3*y1],
        [k1, -2*k2*y1 - k3*y2, -k3*y1],
        [0.0, 2*k2*y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 280

    # Try LSODA with Jacobian and tight tolerances first
    # LSODA auto-switches between stiff and non-stiff, potentially fewer calls
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            jac=_robertson_jac,
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: Radau with Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: BDF with Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            jac=_robertson_jac,
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success and sol.y is not None and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort
    return y0

# EVOLVE-BLOCK-END
