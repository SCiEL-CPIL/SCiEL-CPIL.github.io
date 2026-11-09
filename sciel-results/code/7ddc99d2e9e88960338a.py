"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as si


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an adaptive implicit/explicit strategy for the 100-point convection-diffusion problem.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: evaluator kills after 330s, keep safety margin of ~20s
    deadline = time.time() + 310.0
    
    # Count f calls
    f_call_count = 0
    
    def f_counted(t, y):
        nonlocal f_call_count
        f_call_count += 1
        return f(t, y)
    
    # Strategy 1: Try Radau (implicit, good for stiff problems) with tight tolerances
    # Convection-diffusion with 100 points can be stiff
    try:
        sol = si.solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 50.0:
        # Fallback to simple RK4 with moderate steps
        h = (t1 - t0) / 2000.0
        t, y = t0, y0.copy()
        for _ in range(2000):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    # Strategy 2: Try BDF (implicit, another stiff solver)
    try:
        sol = si.solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30.0:
        # Fallback to RK4
        h = (t1 - t0) / 2000.0
        t, y = t0, y0.copy()
        for _ in range(2000):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    # Strategy 3: Try explicit RK45 with very tight tolerances
    try:
        sol = si.solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='RK45',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 50.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: high-accuracy RK4
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
