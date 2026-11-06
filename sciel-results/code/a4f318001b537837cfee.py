import numpy as np
import time

def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    start_time = time.time()
    for i in range(steps):
        # Deadline check
        if i % 10000 == 0 and time.time() - start_time > 250:
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

def _backward_euler(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    for i in range(steps):
        if i % 100 == 0 and time.time() - start_time > 250:
            return None

        z = y.copy()
        # Newton iterations
        for _newton in range(5):
            fz = f(t + h, z)
            # Finite difference Jacobian
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            
            # Solve (I - h J) delta = -(z - y - h fz)
            A = np.eye(n) - h * J
            b = -(z - y - h * fz)
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use 500,000 steps for RK4 (2,000,000 calls)
    # This is well within the call limit and should be stable for most diffusion coefficients
    y = _rk4(f, t0, t1, y0, 500000)
    if y is None:
        # Fallback to Backward Euler with 2000 steps
        y = _backward_euler(f, t0, t1, y0, 2000)
    return y
