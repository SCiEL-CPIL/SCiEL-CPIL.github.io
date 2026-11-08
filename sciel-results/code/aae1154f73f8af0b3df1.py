"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def _backward_euler(f, t0, t1, y0, steps):
    # Implicit Euler with a finite-difference Jacobian and a few Newton iterations (handles stiff problems).
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        z = y.copy()
        for _newton in range(4):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def integrate(f, t_span, y0):
    # Explicit RK4 first; fall back to implicit Euler when the explicit method blows up (stiff problem).
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0 # 300 seconds limit, keep margin
    
    # Try high-order explicit method first
    # DOP853 is 8th order, very efficient for smooth problems
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1 - t0) / 10.0 # Allow large steps if smooth
        )
        
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If explicit failed or was too slow/unstable, try implicit
    # Radau is good for stiff problems
    try:
        if time.time() > deadline - 30:
             # Fallback to a simple fixed-step RK4 if time is tight
             # This is a last resort
             steps = 1000
             h = (t1 - t0) / steps
             t, y = t0, y0.copy()
             for _ in range(steps):
                 k1 = f(t, y)
                 k2 = f(t + h/2, y + h/2 * k1)
                 k3 = f(t + h/2, y + h/2 * k2)
                 k4 = f(t + h, y + h * k3)
                 y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                 t += h
             return y
             
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Fixed step RK4
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline:
            return y
    return y
# EVOLVE-BLOCK-END
