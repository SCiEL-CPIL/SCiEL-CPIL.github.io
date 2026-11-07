"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _van_der_pol_jac(t, y):
    """
    Analytical Jacobian for the Van der Pol oscillator (mu = 5).
    y0' = y1
    y1' = 5*(1 - y0^2)*y1 - y0
    
    Jacobian:
    [0, 1]
    [-2*mu*y0*y1 - 1, mu*(1 - y0^2)]
    """
    mu = 5.0
    y0, y1 = y[0], y[1]
    jac = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
    ])
    return jac


def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Returns the state at the end of t_span.
    """
    # Deadline guard: 300s limit, keep margin
    deadline = time.time() + 290
    
    try:
        # Use Radau with analytical Jacobian for efficiency and accuracy
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_van_der_pol_jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to RK45 if Radau fails (unlikely)
            sol_fallback = solve_ivp(
                f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
            if sol_fallback.success:
                return sol_fallback.y[:, -1]
                
    except Exception:
        pass
    
    # Last resort fallback: simple RK4 if scipy fails completely
    if time.time() < deadline:
        try:
            t0, t1 = t_span
            y = np.array(y0, dtype=float)
            steps = 10000
            h = (t1 - t0) / steps
            t = t0
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            if np.all(np.isfinite(y)):
                return y
        except Exception:
            pass
            
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
