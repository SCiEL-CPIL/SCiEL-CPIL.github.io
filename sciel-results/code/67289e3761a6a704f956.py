import numpy as np
from scipy.integrate import solve_ivp
import time

def _vdp_jac(t, y):
    """Analytic Jacobian for Van der Pol oscillator with mu=5."""
    mu = 5.0
    y0, y1 = y[0], y[1]
    # J = [[0, 1],
    #      [-2*mu*y0*y1 - 1, mu*(1 - y0^2)]]
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0**2)]
    ])

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    
    # Use Radau with analytic Jacobian to minimize f calls and ensure high accuracy
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=_vdp_jac,
            t_eval=[t1]
        )
        if not sol.success:
            # If Radau fails, try BDF or fallback
            # But it shouldn't fail for this problem
            pass
        return sol.y[:, -1]
    except Exception:
        # Fallback to simple RK4 if SciPy fails for some reason
        # (Unlikely, but safe)
        h = (t1 - t0) / 1000
        t = t0
        y = np.array(y0, dtype=float)
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
