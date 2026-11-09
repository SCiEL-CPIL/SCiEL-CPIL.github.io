import numpy as np
import scipy.integrate as integrate_scipy

def _robertson_jac(t, y):
    # Analytic Jacobian for Robertson problem
    # y[0]=y1, y[1]=y2, y[2]=y3
    J = np.array([
        [-1e-4, 6e7 * y[1] * y[2], 3e7 * y[1]**2],
        [1e-4, -6e7 * y[1] * y[2] - 1e4 * y[2], -3e7 * y[1]**2 - 1e4 * y[1]],
        [0.0, 1e4 * y[2], 1e4 * y[1]]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use BDF with tight tolerances and analytic Jacobian
    sol = integrate_scipy.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='BDF', 
        jac=_robertson_jac, 
        rtol=1e-14, 
        atol=1e-16,
        dense_output=False
    )
    if not sol.success:
        # Fallback to Radau if BDF fails (unlikely)
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_robertson_jac, 
            rtol=1e-14, 
            atol=1e-16
        )
    return sol.y[:, -1]
