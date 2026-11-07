import numpy as np
from scipy.integrate import odeint

def integrate(f, t_span, y0):
    t_start, t_end = t_span
    # odeint requires a sequence of time points
    # We can provide a single point or a dense output?
    # odeint returns the solution at the specified t values.
    # To get high accuracy, we should ensure the internal step size is small.
    # rtol and atol are passed to the solver.
    
    # Note: odeint's interface is slightly different.
    # It integrates from t_start to t_end.
    # We pass t_span as the time grid? No, we can pass a single endpoint?
    # Actually, odeint(y0, t) integrates to all t in t.
    # If we pass t = [t_end], it integrates from t_start to t_end.
    
    y_end = odeint(
        f, 
        y0, 
        np.array([t_end]), 
        args=(), 
        method='lsoda', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    # y_end is shape (1, 2)
    return y_end[0]
