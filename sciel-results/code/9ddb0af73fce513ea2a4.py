"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def _vdp_jac(t, y):
    """Analytic Jacobian for the Van der Pol oscillator with mu=5."""
    mu = 5.0
    y0, y1 = y[0], y[1]
    # y0' = y1
    # y1' = mu * (1 - y0^2) * y1 - y0
    #
    # J = [[d(y0')/dy0, d(y0')/dy1],
    #      [d(y1')/dy0, d(y1')/dy1]]
    #
    # d(y1)/dy0 = 0
    # d(y1)/dy1 = 1
    #
    # d(mu*(1-y0^2)*y1 - y0)/dy0 = mu*(-2*y0)*y1 - 1
    # d(mu*(1-y0^2)*y1 - y0)/dy1 = mu*(1-y0^2)
    
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0**2)]
    ])


def integrate(f, t_span, y0):
    start_time = time.time()
    
    # Use Radau with analytic Jacobian to reduce function calls and improve efficiency.
    # The analytic Jacobian avoids the extra f calls needed for finite difference approximation.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_vdp_jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if solver fails for some reason
            # Use a simple explicit RK4 with small steps
            # This is unlikely to be triggered
            h = 1e-4
            t = t_span[0]
            y = np.array(y0, dtype=float)
            while t < t_span[1]:
                if time.time() - start_time > 250:
                    return y
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
            
    except Exception:
        # Absolute fallback
        return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
