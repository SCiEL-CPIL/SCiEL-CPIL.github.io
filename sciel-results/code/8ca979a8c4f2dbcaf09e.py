import numpy as np
from scipy.integrate import solve_ivp

def _robertson_jac(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    """
    y0, y1, y2 = y
    # Ensure y is 1D array
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e7 * y1, -1e4 * y1],
        [0.0, 6e7 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem.
    """
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau for stiff problems
    # rtol and atol set to high precision
    # Provide analytic Jacobian to minimize RHS function calls
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if not sol.success:
            # Fallback if Radau fails
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                jac=_robertson_jac,
                rtol=1e-12,
                atol=1e-14
            )
        return sol.y[:, -1]
    except Exception:
        # Last resort: simple RK4 with many steps (unlikely to be needed)
        # But Radau should work
        import warnings
        warnings.warn("Radau/BDF failed, falling back to RK4")
        t0, t1 = t_span
        h = (t1 - t0) / 10000
        t, y = t0, y0.copy()
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
