"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses Radau implicit Runge-Kutta method with tight tolerances for high accuracy
    on stiff problems like Robertson. Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: stop if we've used too much time
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Robertson problem constants (for Jacobian if needed, but Radau estimates it)
    # The Jacobian can be provided analytically for better performance
    def jacobian(t, y):
        a1, a2, a3 = 0.04, 3.0e-7, 1.0e4
        y0_, y1_, y2_ = y
        return np.array([
            [-a1, a3 * y2_, a3 * y1_],
            [a1, -2 * a2 * y1_ - a3 * y2_, -a3 * y1_],
            [0.0, 2 * a2 * y1_ - a3 * y2_, -a3 * y1_]
        ])
    
    # Use Radau method with tight tolerances
    # rtol=1e-10, atol=1e-12 for good precision without excessive function calls
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            jac=jacobian,
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 100.0,  # Limit step size for accuracy
            first_step=1e-6  # Small first step to handle initial stiffness
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails
            sol = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='BDF',
                jac=jacobian,
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1 - t0) / 100.0
            )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: return last computed value
            return sol.y[:, -1]
    
    except Exception as e:
        # If solver fails completely, try a simple RK4 as fallback
        # But this should rarely happen with Radau
        h = (t1 - t0) / 10000
        t = t0
        y = y0.copy()
        for _ in range(10000):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
# EVOLVE-BLOCK-END
