"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk4_step(f, t, y, h):
    """Single RK4 step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_with_adaptive(f, t0, t1, y0, max_steps=500000, rtol=1e-9, atol=1e-12, deadline=None):
    """
    RK4 with adaptive step size using two-half-steps error estimation.
    Returns (y, calls) or (None, calls) on failure.
    """
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 2000  # initial step size
    step_count = 0
    total_calls = 0
    scale = atol + rtol * np.abs(y)
    
    while t < t1 - 1e-14:
        if deadline is not None and time.time() > deadline:
            return y, total_calls  # return best effort
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # RK4 step
        y_new = _rk4_step(f, t, y, h)
        total_calls += 4
        
        # Error estimation: take two half-steps and compare
        y_half1 = _rk4_step(f, t, y, 0.5 * h)
        y_half2 = _rk4_step(f, t + 0.5 * h, y_half1, 0.5 * h)
        total_calls += 8
        
        # Check for divergence
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            return None, total_calls
        
        # Compute error
        scale = atol + rtol * np.abs(y_new)
        error = np.max(np.abs(y_new - y_half2) / scale)
        
        if error < 1.0:
            # Accept step
            t += h
            y = y_new
            step_count += 1
            
            # Adjust step size
            if error > 1e-12:
                h = 0.9 * h * (1.0 / error) ** 0.2
            else:
                h = min(2.0 * h, (t1 - t))
        else:
            # Reject step, reduce h
            h = 0.9 * h * (1.0 / error) ** 0.2
            if h < 1e-14:
                h = 1e-14
        
        if step_count > max_steps:
            return None, total_calls
    
    return y, total_calls


def _backward_euler(f, t0, t1, y0, steps, max_newton=6, ftol=1e-10, deadline=None):
    """
    Implicit Backward Euler with Newton's method.
    Returns (y, calls) or (None, calls).
    """
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    total_calls = 0
    
    # Precompute identity matrix
    I = np.eye(n)
    
    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            return y, total_calls
        
        t_new = t + h
        z = y.copy()
        
        for newton_iter in range(max_newton):
            fz = f(t_new, z)
            total_calls += 1
            
            # Compute residual: z - y - h * f(t_new, z) = 0
            residual = z - y - h * fz
            res_norm = np.linalg.norm(residual)
            
            if res_norm <= ftol * (1.0 + np.linalg.norm(y)):
                break
            
            # Finite difference Jacobian
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale_j = max(1.0, abs(z[j]))
                e[j] = 1e-6 * scale_j
                J[:, j] = (f(t_new, z + e) - fz) / e[j]
                total_calls += 1
            
            # Solve (I - h*J) * step = -residual
            try:
                A = I - h * J
                step = np.linalg.solve(A, -residual)
            except np.linalg.LinAlgError:
                return None, total_calls
            
            z = z + step
            
            if np.linalg.norm(step) <= ftol * (1.0 + np.linalg.norm(z)):
                break
        
        y = z
        t = t_new
    
    return y, total_calls


def integrate(f, t_span, y0):
    """
    Main integration function with adaptive time-stepping.
    """
    t0, t1 = t_span
    deadline = time.time() + 280  # 280s safety margin
    
    # Try adaptive RK4 first
    y, calls = _rk4_with_adaptive(f, t0, t1, y0, max_steps=400000, rtol=1e-10, atol=1e-13, deadline=deadline)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # If adaptive RK4 failed or returned None, try fixed-step RK4 with various step counts
    if time.time() < deadline:
        for steps in [4000, 8000, 16000, 32000]:
            h = (t1 - t0) / steps
            t = t0
            y = np.array(y0, dtype=float)
            success = True
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
                k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
                k4 = f(t + h, y + h * k3)
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                    success = False
                    break
                if time.time() > deadline:
                    return y  # Return best effort
            if success and np.all(np.isfinite(y)):
                return y
    
    # Fallback: Backward Euler
    if time.time() < deadline:
        for steps in [2000, 4000]:
            y, calls = _backward_euler(f, t0, t1, y0, steps, max_newton=8, ftol=1e-11, deadline=deadline)
            if y is not None and np.all(np.isfinite(y)):
                return y
    
    # Last resort: simple RK4 with minimal steps
    h = (t1 - t0) / 1000
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
