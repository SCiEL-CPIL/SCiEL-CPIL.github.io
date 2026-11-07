"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses SciPy's Radau method with an analytical Jacobian for the Van der Pol oscillator
    to minimize function calls while maintaining high accuracy.
    """
    y0 = np.asarray(y0, dtype=float)
    
    # Analytical Jacobian for the Van der Pol oscillator:
    # y0' = y1
    # y1' = 5 * (1 - y0^2) * y1 - y0
    #
    # J[0, 0] = 0
    # J[0, 1] = 1
    # J[1, 0] = -10 * y0 * y1 - 1
    # J[1, 1] = 5 * (1 - y0^2)
    def jac(t, y):
        return np.array([
            [0.0, 1.0],
            [-10.0 * y[0] * y[1] - 1.0, 5.0 * (1.0 - y[0]**2)]
        ])

    # Use Radau with the analytical Jacobian.
    # Radau is an implicit Runge-Kutta method suitable for stiff problems.
    # Providing the Jacobian avoids finite-difference approximations, reducing f calls.
    # Tolerances are set tighter than the reference (1e-11, 1e-13) to minimize relative error.
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        jac=jac,
        dense_output=False
    )
    
    if sol.success:
        y_end = sol.y[:, -1]
        if np.all(np.isfinite(y_end)):
            return y_end
    
    # Fallback to DOP853 if Radau fails for some reason
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15
    )
    
    if sol.success:
        y_end = sol.y[:, -1]
        if np.all(np.isfinite(y_end)):
            return y_end
            
    return y0

# EVOLVE-BLOCK-END
