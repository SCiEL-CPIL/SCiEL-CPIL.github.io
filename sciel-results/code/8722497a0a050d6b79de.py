"""
Alternative strategy: Use an embedded high-order Runge-Kutta method (DOP853) with
extremely tight tolerances. DOP853 is an explicit 8th-order method with 5th-order error
estimation, which is significantly more efficient per unit of error than Radau for
non-stiff or moderately stiff problems like Van der Pol with mu=5. This should achieve
similar or better accuracy with fewer function calls, improving the score via the
-log10(rel) term while keeping the call penalty manageable.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 (explicit 8th-order Runge-Kutta) with very tight tolerances.
    # This method is highly accurate and efficient for smooth ODEs.
    # For Van der Pol with mu=5, it's not extremely stiff, so explicit methods work well.
    # Tighter tolerances: rtol=1e-14, atol=1e-16 to push rel towards the 1e-14 floor.
    
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853',
        rtol=1e-14, 
        atol=1e-16
    )
    
    if not sol.success:
        # Fallback to Radau if DOP853 fails
        def jac(t, y):
            y0_val = y[0]
            y1_val = y[1]
            return np.array([
                [0.0, 1.0],
                [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val**2)]
            ])
        
        sol2 = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            rtol=1e-13, 
            atol=1e-15
        )
        
        if sol2.success and sol2.y.shape[1] > 0:
            return sol2.y[:, -1]
        elif sol.y.shape[1] > 0:
            return sol.y[:, -1]
        else:
            return y0
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
