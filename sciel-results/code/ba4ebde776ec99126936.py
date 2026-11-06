"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """Single Dormand-Prince RK45 step. Returns y4 (4th order) and y5 (5th order)."""
    k1 = f(t, y)
    k2 = f(t + h / 5.0, y + h * (k1 / 5.0))
    k3 = f(t + 3.0 * h / 10.0, y + h * (3.0 * k1 / 40.0 + 9.0 * k2 / 40.0))
    k4 = f(t + 4.0 * h / 5.0, y + h * (44.0 * k1 / 45.0 - 5.0 * k2 + 56.0 * k3 / 15.0))
    k5 = f(t + h, y + h * (-11.0 * k1 / 54.0 + 5.0 * k2 - 70.0 * k3 / 27.0 + 35.0 * k4 / 27.0))
    k6 = f(t + h / 2.0, y + h * (1631.0 * k1 / 55296.0 + 175.0 * k2 / 512.0 + 575.0 * k3 / 13824.0 + 44275.0 * k4 / 110592.0 - 512.0 * k5 / 175.0))
    
    # 5th order solution
    y5 = y + h * (361.0 * k1 / 5040.0 + 200.0 * k3 / 427.0 + 280.0 * k4 / 405.0 + 127.0 * k5 / 40.0 - k6)
    # 4th order solution
    y4 = y + h * (1951.0 * k1 / 21600.0 + 2277.0 * k3 / 26880.0 + 67.0 * k4 / 45.0 + 2825.0 * k5 / 2688.0 - 11.0 * k6 / 4.0)
    return y4, y5


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=500000, deadline=None):
    """Adaptive Dormand-Prince RK45 with step size control."""
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimate
    h = (t1 - t0) / 100.0
    h_min = 1e-12
    h_max = (t1 - t0) / 10.0
    
    step_count = 0
    while t < t1 - 1e-15:
        if deadline is not None and time.time() > deadline:
            return y, step_count
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min
        
        y4, y5 = _rk45_step(f, t, y, h)
        step_count += 6  # 6 function evaluations per step
        
        # Error estimate
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.max(err / scale)
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            # Adjust step size
            if err_norm > 1e-15:
                h_new = h * min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, h_max)
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * err_norm ** (-0.25))
        
        if step_count > max_steps:
            break
    
    return y, step_count


def _implicit_euler(f, t0, t1, y0, steps=2000, deadline=None):
    """Implicit Euler with Newton iteration for stiff problems."""
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    I = np.eye(n)
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            return y
        
        t_new = t + h
        z = y.copy()
        fz = f(t_new, z)
        
        # Compute Jacobian using finite differences
        J = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-8 * max(1.0, abs(z[j]))
            J[:, j] = (f(t_new, z + e) - fz) / e[j]
        
        # Solve (I - h*J) dz = -(z - y - h*fz) for Newton step
        A = I - h * J
        b = -(z - y - h * fz)
        
        for _newton in range(5):
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                break
            z = z + step
            fz_new = f(t_new, z)
            b = -(z - y - h * fz_new)
            if np.linalg.norm(step) <= 1e-10 * (1.0 + np.linalg.norm(z)):
                break
            # Recompute Jacobian occasionally
            if _newton == 2:
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = 1e-8 * max(1.0, abs(z[j]))
                    J[:, j] = (f(t_new, z + e) - fz_new) / e[j]
                A = I - h * J
        
        y = z
        t = t_new
    
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Try adaptive RK45 first
    y, evals = _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12, deadline=deadline)
    
    # Check if we made progress
    if np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e12:
        return y
    
    # Fallback to implicit Euler
    y = _implicit_euler(f, t0, t1, y0, steps=2000, deadline=deadline)
    return y
# EVOLVE-BLOCK-END
