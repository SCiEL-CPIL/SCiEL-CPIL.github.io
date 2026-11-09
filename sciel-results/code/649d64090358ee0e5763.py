import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using scipy.integrate.solve_ivp
    with DOP853 method to match the reference solution's high accuracy
    and efficiency.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if solve_ivp fails
            return _fallback_integrate(f, t_span, y0)
    except Exception:
        return _fallback_integrate(f, t_span, y0)

def _fallback_integrate(f, t_span, y0):
    # Simple high-step RK4 as a last resort
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
    return y
