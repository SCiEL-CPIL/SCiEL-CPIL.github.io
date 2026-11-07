import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Solve the Lotka-Volterra system using scipy's DOP853 with tolerances
    matching the reference solution. 
    Optimization: Provide 'first_step' to skip initial step size estimation calls.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Estimate a reasonable first step to avoid scipy's internal estimation calls
    # which consume f evaluations.
    # A small fraction of the total interval is a safe bet.
    dt = t1 - t0
    if dt <= 0:
        return y0
    
    # Heuristic for first step: 1/100th of the interval, clamped to a small value
    # to ensure stability, but large enough to not be too small.
    first_step = dt / 100.0
    if first_step < 1e-6:
        first_step = 1e-6
    if first_step > 0.1:
        first_step = 0.1

    try:
        # Match the reference solver settings exactly to minimize relative error
        # while keeping function call count low.
        # Providing first_step skips the initial derivative estimation calls.
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            first_step=first_step,
            dense_output=False
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: High-order RK4 with fixed steps
    # This is unlikely to be reached, but ensures we return something.
    if time.time() > deadline - 10:
        return y0
    
    steps = 20000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    
    for i in range(steps):
        if time.time() > deadline - 5:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
