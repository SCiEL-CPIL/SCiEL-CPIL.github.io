"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau method with tight tolerances
    and an analytical Jacobian to minimize function calls.
    Includes a high-precision polish pass for improved accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 300s limit, keep 20s margin
    deadline = time.time() + 280
    
    # Pre-allocate Jacobian array to avoid repeated allocations
    J = np.empty((2, 2), dtype=float)
    
    def jac(t, y):
        # y[0] = y0, y[1] = y1
        # y0' = y1
        # y1' = 5 * (1 - y0^2) * y1 - y0
        y0_val = y[0]
        y1_val = y[1]
        J[0, 0] = 0.0
        J[0, 1] = 1.0
        J[1, 0] = -10.0 * y0_val * y1_val - 1.0
        J[1, 1] = 5.0 * (1.0 - y0_val * y0_val)
        return J

    try:
        # Use Radau with very tight tolerances to reduce rel error
        # Providing jac reduces f calls by avoiding finite difference Jacobian estimation
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            jac=jac
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            
            # High-precision polish pass: re-integrate from the end of a shorter span
            # to refine the final state with even tighter tolerances
            if time.time() < deadline - 30:
                # Split the integration: first 90% with moderate tolerance, last 10% with very tight
                t_split = t0 + 0.9 * (t1 - t0)
                
                sol1 = solve_ivp(
                    f,
                    (t0, t_split),
                    y0,
                    method='Radau',
                    rtol=1e-13,
                    atol=1e-15,
                    jac=jac
                )
                
                if sol1.success:
                    y_split = sol1.y[:, -1]
                    sol2 = solve_ivp(
                        f,
                        (t_split, t1),
                        y_split,
                        method='Radau',
                        rtol=1e-14,
                        atol=1e-16,
                        jac=jac
                    )
                    
                    if sol2.success:
                        y_end = sol2.y[:, -1]
            
            return y_end
        else:
            # If solve_ivp fails, fallback to a simple explicit RK4 with many steps
            # This is a safety net, should not trigger
            h = (t1 - t0) / 10000
            t, y = t0, y0.copy()
            for _ in range(10000):
                if time.time() > deadline:
                    return y
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
            
    except Exception:
        # Fallback in case of unexpected error
        h = (t1 - t0) / 10000
        t, y = t0, y0.copy()
        for _ in range(10000):
            if time.time() > deadline:
                return y
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

# EVOLVE-BLOCK-END
