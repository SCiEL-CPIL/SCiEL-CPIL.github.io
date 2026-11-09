import numpy as np
from scipy.integrate import odeint

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # LSODA is an adaptive BDF/Adams method, good for stiff problems
    # It is a different approach family than Radau (Implicit RK)
    
    # Wrapper to match odeint's expected signature: func(y, t, *args)
    # where args can include the original f
    def func(y, t, f_func):
        return f_func(t, y)
    
    # We only need the final state, so t array is [t0, t1]
    # odeint integrates from t[0] to t[-1]
    t = np.array([t0, t1])
    
    try:
        sol = odeint(func, y0, t, args=(f,), integrator='lsoda', rtol=1e-10, atol=1e-12)
        return sol[-1]
    except Exception:
        # Fallback to BDF in solve_ivp if LSODA fails for some reason
        from scipy.integrate import solve_ivp
        sol2 = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-10, atol=1e-12)
        if sol2.success:
            return sol2.y[:, -1]
        return sol2.y[:, -1] if sol2.y.size > 0 else np.array(y0, dtype=float)
