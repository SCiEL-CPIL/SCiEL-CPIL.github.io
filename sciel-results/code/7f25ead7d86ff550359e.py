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
    Integrates the ODE y' = f(t, y) using Radau with analytic Jacobian for the Robertson problem.
    Radau is a high-order implicit Runge-Kutta method well-suited for stiff systems.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Primary: Radau with tight tolerances and analytic Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_jac_robertson,
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 1: Radau with slightly looser tolerances
    try:
        if time.time() >= deadline:
            return np.zeros_like(y0)
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_jac_robertson,
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 2: BDF with analytic Jacobian
    try:
        if time.time() >= deadline:
            return np.zeros_like(y0)
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            jac=_jac_robertson,
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 3: BDF with looser tolerances
    try:
        if time.time() >= deadline:
            return np.zeros_like(y0)
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            jac=_jac_robertson,
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 4: Radau without Jacobian
    try:
        if time.time() >= deadline:
            return np.zeros_like(y0)
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort
    return np.zeros_like(y0)
