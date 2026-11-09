import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses scipy.integrate.solve_ivp with the DOP853 method (8th order explicit Runge-Kutta)
    for high accuracy and efficiency.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # DOP853 is suitable for non-stiff problems and provides high accuracy.
    # Tight tolerances ensure the result is close to the reference solution.
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        # max_step prevents the solver from taking a step that is too large
        # relative to the problem scale, though adaptive control usually handles this.
        # We set it to a fraction of the total span to ensure sufficient resolution.
        max_step=(t1 - t0) / 20 
    )
    
    return sol.y[:, -1]
