"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as si
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Primary strategy: exploit linearity (y' = A y) via matrix exponential.
    Fallback: Radau / DOP853 / RK4 for non-linear or time-dependent cases.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Deadline guard: evaluator kills after 330s, keep safety margin of ~30s
    deadline = time.time() + 290.0

    # Strategy 1: Attempt to detect and solve linear autonomous ODE y' = A y
    # This is highly efficient for convection-diffusion (linear, time-invariant).
    try:
        # Construct A by evaluating f at t0 for basis vectors
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            A[:, j] = f(t0, ej)
            
        # Verify linearity and time-invariance with a few sample points
        # Check 1: f(t0, y0) == A y0
        pred1 = A @ y0
        actual1 = f(t0, y0)
        if np.max(np.abs(pred1 - actual1)) > 1e-8:
            raise ValueError("Not linear at t0")
            
        # Check 2: f(t_mid, y0) == A y0 (time invariance)
        t_mid = 0.5 * (t0 + t1)
        actual2 = f(t_mid, y0)
        if np.max(np.abs(pred1 - actual2)) > 1e-8:
            raise ValueError("Time-dependent")
            
        # Check 3: f(t0, y0 * 2) == A (y0 * 2) (linearity)
        actual3 = f(t0, y0 * 2.0)
        pred3 = A @ (y0 * 2.0)
        if np.max(np.abs(pred3 - actual3)) > 1e-8:
            raise ValueError("Non-linear")
            
        # If all checks pass, compute solution via matrix exponential
        M = expm(A * dt)
        y_end = M @ y0
        return y_end
        
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - 100.0:
        # Fallback to simple RK4
        steps = 5000
        h = dt / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    # Strategy 2: Radau (implicit, good for stiff problems)
    try:
        sol = si.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            max_step=dt / 100.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - 50.0:
        steps = 5000
        h = dt / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    # Strategy 3: DOP853 (high-order explicit)
    try:
        sol = si.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-13, 
            atol=1e-15,
            max_step=dt / 50.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: high-accuracy RK4
    steps = 10000
    h = dt / steps
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
