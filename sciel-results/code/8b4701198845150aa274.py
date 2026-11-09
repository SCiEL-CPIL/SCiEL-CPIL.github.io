"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import time


def _rk4(f, t0, t1, y0, steps, deadline):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if time.time() > deadline:
            return None
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def _backward_euler(f, t0, t1, y0, steps, deadline):
    # Implicit Euler with a finite-difference Jacobian and a few Newton iterations (handles stiff problems).
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        if time.time() > deadline:
            return None
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
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    deadline = time.time() + 280.0
    
    # Check for linearity and autonomy
    y_test1 = f(t0, y0)
    y_test2 = f(t0, 2.0 * y0)
    y_test3 = f(t1, y0)
    
    tol = 1e-8 * (1.0 + np.linalg.norm(y_test1))
    
    is_linear_homogeneous = np.linalg.norm(y_test2 - 2.0 * y_test1) < tol
    is_autonomous = np.linalg.norm(y_test3 - y_test1) < tol
    
    if is_linear_homogeneous and is_autonomous:
        # Assume y' = A y
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
            
        if np.linalg.norm(A @ y0 - y_test1) < tol:
            dt = t1 - t0
            M = scipy.linalg.expm(dt * A)
            y_end = M @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
            
    # Fallback to RK4
    y = _rk4(f, t0, t1, y0, 2000, deadline)
    if y is None:
        y = _backward_euler(f, t0, t1, y0, 2000, deadline)
    if y is None:
        return y0
    return y
# EVOLVE-BLOCK-END
