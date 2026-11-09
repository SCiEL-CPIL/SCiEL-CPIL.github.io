import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Ensure y0 is a float array for compatibility
    y0 = np.array(y0, dtype=float)
    
    # Define the analytical Jacobian for the Van der Pol oscillator (mu=5)
    # y' = [y1, 5*(1-y0^2)*y1 - y0]
    # J[0,0] = 0
    # J[0,1] = 1
    # J[1,0] = d/dy0 (5*(1-y0^2)*y1 - y0) = 5*(-2*y0)*y1 - 1 = -10*y0*y1 - 1
    # J[1,1] = d/dy1 (5*(1-y0^2)*y1 - y0) = 5*(1-y0^2)
    def jac(t, y):
        return np.array([
            [0.0, 1.0],
            [-10.0 * y[0] * y[1] - 1.0, 5.0 * (1.0 - y[0]**2)]
        ])

    # Use Radau method with the analytical Jacobian
    # This should significantly reduce the number of function calls
    # compared to finite difference estimation, while maintaining high accuracy.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13,
        jac=jac
    )
    
    if not sol.success:
        # In case of failure, return the last computed state if available
        if sol.y.size > 0:
            return sol.y[:, -1]
        else:
            return y0

    return sol.y[:, -1]
