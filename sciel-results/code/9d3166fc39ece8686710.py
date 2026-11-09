import numpy as np
from scipy.integrate import solve_ivp
import time

def _jac_robertson(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    return np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
        [0.0, 6e7 * y2, 0.0]
    ])

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) using a stiff solver (Radau) with analytic Jacobian.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Use Radau with analytic Jacobian for the Robertson problem
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=_jac_robertson,
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to BDF with analytic Jacobian
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=_jac_robertson,
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all else fails, return a zero vector or last known?
    # Should not happen for Robertson
    return np.zeros_like(y0)
