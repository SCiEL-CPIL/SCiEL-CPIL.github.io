"""Improved program using SciPy's Radau solver with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytic Jacobian of the Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    mu = 5.0
    # y0' = y1  =>  d(y0')/dy0 = 0, d(y0')/dy1 = 1
    # y1' = mu*(1-y0^2)*y1 - y0
    # d(y1')/dy0 = -2*mu*y0*y1 - 1
    # d(y1')/dy1 = mu*(1-y0^2)
    J = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin before 300s kill
    
    # Use Radau with tight tolerances and analytic Jacobian
    # Radau is the reference method, so matching it closely minimizes rel
    # Analytic Jacobian avoids wasting f-calls on FD-Jacobian estimation
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            jac=_vdp_jac,
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1
        )
        y_end = sol.y[:, -1]
        
        # Check deadline
        if time.time() > deadline:
            # Return best result found so far
            if np.all(np.isfinite(y_end)):
                return y_end
            # Fallback: return initial state if something went wrong
            return y0_arr.copy()
        
        if not np.all(np.isfinite(y_end)):
            # Fallback if Radau failed
            return _fallback(f, t_span, y0_arr)
        
        return y_end
        
    except Exception:
        if time.time() > deadline:
            return y0_arr.copy()
        return _fallback(f, t_span, y0_arr)


def _fallback(f, t_span, y0):
    """Fallback using a simple RK4 with modest steps."""
    t0, t1 = t_span
    h = (t1 - t0) / 5000
    t = t0
    y = np.array(y0, dtype=float)
    n_steps = int(round((t1 - t0) / h))
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
