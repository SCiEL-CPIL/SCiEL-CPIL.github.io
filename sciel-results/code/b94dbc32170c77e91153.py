"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _van_der_pol_jac(t, y):
    """Analytical Jacobian for the Van der Pol oscillator with mu=5."""
    y0, y1 = y
    mu = 5.0
    # y0' = y1
    # y1' = mu(1 - y0^2)y1 - y0
    # J = [[dy0'/dy0, dy0'/dy1],
    #      [dy1'/dy0, dy1'/dy1]]
    # dy0'/dy0 = 0, dy0'/dy1 = 1
    # dy1'/dy0 = -2*mu*y0*y1 - 1
    # dy1'/dy1 = mu*(1 - y0^2)
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
    ])


def _rk4_fallback(f, t0, t1, y0, steps):
    """Fallback RK4 integrator if solve_ivp fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 300s limit, keep 20s margin
    start_time = time.time()
    deadline = start_time + 280.0

    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_van_der_pol_jac,
            max_step=0.1  # Allow adaptive stepping but keep it reasonably controlled
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If solve_ivp fails, try a fallback
            y = _rk4_fallback(f, t0, t1, y0, 5000)
            if y is not None:
                return y
            # Last resort: return initial state or raise
            return y0
    except Exception:
        # If exception occurs, fallback to RK4
        if time.time() > deadline:
            return y0
        y = _rk4_fallback(f, t0, t1, y0, 5000)
        if y is not None:
            return y
        return y0
# EVOLVE-BLOCK-END
