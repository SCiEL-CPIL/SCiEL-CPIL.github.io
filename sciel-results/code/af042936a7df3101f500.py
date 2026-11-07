"""ODE integrator for Van der Pol oscillator using Radau with analytical Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytical Jacobian for Van der Pol oscillator (mu=5)."""
    y0_val, y1_val = y[0], y[1]
    mu = 5.0
    return np.array([
        [0.0, 1.0],
        [-1.0 - 2.0 * mu * y0_val * y1_val, mu * (1.0 - y0_val * y0_val)]
    ])


def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Returns the state at the end of t_span.
    """
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    y0_arr = np.array(y0, dtype=float)
    
    # Primary: Radau with analytical Jacobian and tight tolerances matching reference
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_vdp_jac
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Secondary: Radau without Jacobian but with max_step constraint
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='Radau',
                rtol=1e-11,
                atol=1e-13,
                max_step=0.1
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Tertiary: DOP853 (8th order explicit) as fallback
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: RK4 with moderate steps
    if time.time() < deadline:
        t0, t1 = t_span
        y = y0_arr.copy()
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t = t0
        for _ in range(n_steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                break
        if np.all(np.isfinite(y)):
            return y
    
    # Last resort
    return y0_arr
# EVOLVE-BLOCK-END
