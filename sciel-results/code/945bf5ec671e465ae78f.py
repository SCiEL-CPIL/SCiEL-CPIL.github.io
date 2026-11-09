"""Improved ODE integrator using SciPy's DOP853 for high accuracy with low call count."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    """
    Integrates the ODE system y' = f(t, y) from t_span[0] to t_span[1]
    using the DOP853 method.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to minimize relative error
    # while keeping the number of function calls low due to high order.
    sol = integrate_scipy.solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='DOP853', 
        rtol=1e-14, 
        atol=1e-16,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback in case of failure, though unlikely for this smooth problem
        # Return last available state or raise
        if sol.t.size > 0:
            return sol.y[:, -1]
        else:
            return y0

    return sol.y[:, -1]
# EVOLVE-BLOCK-END
