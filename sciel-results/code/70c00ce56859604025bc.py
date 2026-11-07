import numpy as np
from scipy.integrate import odeint

def integrate(f, t_span, y0):
    t0, t1 = t_span
    t = np.array([t0, t1])
    
    def f_wrapped(y, t):
        return f(t, y)
        
    try:
        # odeint uses LSODA, a different approach family than solve_ivp's DOP853.
        # Using tolerances matching the reference to balance accuracy and call count.
        y_end = odeint(f_wrapped, y0, t, args=(), rtol=1e-11, atol=1e-13, mxstep=10000)
        return y_end[-1]
    except Exception:
        # Fallback: Fixed-step RK4
        y = np.array(y0, dtype=float)
        steps = 10000
        h = (t1 - t0) / steps
        for i in range(steps):
            t = t0 + i * h
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y
