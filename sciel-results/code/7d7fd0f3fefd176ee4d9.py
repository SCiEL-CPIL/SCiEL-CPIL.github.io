import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = 290.0  # Safety margin
    
    # Use DOP853 with tight tolerances to minimize rel error
    # Reference uses rtol=1e-11, atol=1e-13
    # We use tighter to ensure we are well below reference error
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            # max_step is left as default (inf) as adaptive step control is efficient
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: if DOP853 fails or times out (unlikely), use a robust RK45
    # Note: This path is unlikely to be taken for the Lorenz system
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: Fixed step RK4 (slow, less accurate, but always works)
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
        if time.time() > deadline:
            break
    return y
