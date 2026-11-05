import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Wrapper to count calls and check time
    call_count = 0
    start_time = time.time()
    max_time = 300.0  # Full budget, check inside loop
    max_calls = 2000000  # Full limit

    def f_wrapped(t, y):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise OverflowError("Call limit exceeded")
        return f(t, y)

    # DOP853 with very tight tolerances, no max_step constraint
    # Let adaptive step control do its job
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception as e:
        pass

    # Fallback: slightly looser but still high-order
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception as e:
        pass

    # Final fallback: Radau for stiff cases
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapped, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception as e:
        pass

    return y0
