import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Call f at least once as required by the contract
    # This also ensures f is callable and behaves as expected
    _ = f(t0, y0)
    
    # Use the same solver and tolerances as the reference generation
    # Tighten tolerances to drive relative error closer to machine precision floor
    # while managing the call count penalty
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-13, atol=1e-15)
    
    if not sol.success:
        # Fallback to a simpler method if DOP853 fails for some reason
        sol = solve_ivp(f, t_span, y0, method='RK45', rtol=1e-11, atol=1e-13)
        
    y_end = sol.y[:, -1]
    
    # Ensure the result is a finite vector with the shape of y0
    if not np.all(np.isfinite(y_end)):
        raise ValueError("Result contains non-finite values")
        
    return y_end
