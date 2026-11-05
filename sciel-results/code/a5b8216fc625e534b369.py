"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as la

# Global deadline guard: stop integrating and return best estimate if we approach 330s.
_DEADLINE = 325.0  # safety margin before 330s kill


def _get_matrix_at(f, t, y0, n):
    """
    Try to detect if the system is linear time-invariant: y' = A*y + b.
    We evaluate f at y=0 to get b, and at unit vectors to get columns of A.
    """
    y_zero = np.zeros(n, dtype=float)
    b = np.array(f(t, y_zero), dtype=float)
    
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = np.array(f(t, e), dtype=float) - b
    
    return A, b


def _is_linear_time_invariant(f, t, y0, A, b, n, tol=1e-10):
    """
    Check if the system is linear and time-invariant by testing at a few random points.
    """
    # Test at y0
    y_test = np.array(f(t, y0), dtype=float)
    pred = A @ y0 + b
    if not np.allclose(y_test, pred, rtol=tol, atol=tol):
        return False
    
    # Test at a random point
    rng = np.random.default_rng(42)
    y_rand = rng.standard_normal(n)
    y_test2 = np.array(f(t, y_rand), dtype=float)
    pred2 = A @ y_rand + b
    if not np.allclose(y_test2, pred2, rtol=tol, atol=tol):
        return False
    
    return True


def _solve_linear_system(f, t_span, y0):
    """
    Solve a linear time-invariant system y' = A*y + b using matrix exponential.
    """
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Get A and b at t0
    A, b = _get_matrix_at(f, t0, y0, n)
    
    # Check if it's linear time-invariant
    if not _is_linear_time_invariant(f, t0, y0, A, b, n):
        return None
    
    # For linear time-invariant system y' = A*y + b:
    # y(t) = exp(A*(t-t0)) * y0 + integral from 0 to (t-t0) of exp(A*s) * b ds
    # = exp(A*(t-t0)) * y0 + A^{-1} * (exp(A*(t-t0)) - I) * b  (if A is invertible)
    # Or more generally: y(t) = exp(A*(t-t0)) * (y0 + A^{-1}*b) - A^{-1}*b
    
    dt = t1 - t0
    
    # Compute exp(A*dt)
    try:
        exp_A_dt = la.expm(A * dt)
    except Exception:
        return None
    
    # Compute the particular solution
    # If A is invertible: y_p = -A^{-1} * b
    # y(t) = exp(A*dt) * (y0 - A^{-1}*b) + A^{-1}*b
    
    try:
        # Try to solve A * x = b for x = A^{-1} * b
        x = la.solve(A, b)
        y_particular = -x
        y_final = exp_A_dt @ (y0 + y_particular) - y_particular
    except la.LinAlgError:
        # A is singular, use the integral form
        # y(t) = exp(A*dt) * y0 + integral from 0 to dt of exp(A*s) * b ds
        # Approximate the integral using a few quadrature points
        num_quad = 10
        s_vals = np.linspace(0, dt, num_quad + 1)
        integral = np.zeros(n, dtype=float)
        for i in range(num_quad):
            s_mid = 0.5 * (s_vals[i] + s_vals[i+1])
            exp_A_s = la.expm(A * s_mid)
            integral += (s_vals[i+1] - s_vals[i]) * exp_A_s @ b
        y_final = exp_A_dt @ y0 + integral
    
    return y_final


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=50000):
    """
    Adaptive RK45 (Dormand-Prince) with step size control.
    """
    y = np.array(y0, dtype=float)
    n = len(y)
    t = t0
    
    # Initial step size estimate
    h = 1e-4
    
    # Dormand-Prince coefficients
    A = np.array([
        0, 0, 0, 0, 0, 0,
        1/5, 0, 0, 0, 0,
        3/40, 9/40, 0, 0, 0,
        44/45, -56/15, 32/9, 0, 0,
        19372/6561, -25360/2187, 64448/6561, -212/729, 0,
        9017/3168, -355/33, 46732/5247, 49/176, -5103/18656,
        35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    ]).reshape(7, 7)
    
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    
    steps = 0
    while t < t1 - 1e-15:
        if steps >= max_steps:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = np.zeros((7, n), dtype=float)
        for i in range(7):
            y_stage = y + h * np.sum(A[i, :i] * k[:i], axis=0)
            k[i] = f(t + c[i] * h, y_stage)
        
        # 4th order solution
        y_new = y + h * np.sum(b * k, axis=0)
        # 5th order solution
        y_new5 = y + h * np.sum(b_star * k, axis=0)
        
        # Error estimate
        err = np.max(np.abs(y_new5 - y_new) / (atol + rtol * np.abs(y_new)))
        
        if err <= 1.0:
            # Accept step
            t += h
            y = y_new5
            steps += 1
            
            # Adjust step size
            if err < 1e-3:
                h *= min(5.0, 0.9 * err ** (-0.2))
            else:
                h *= min(5.0, 0.9 * err ** (-0.25))
            h = min(h, (t1 - t) * 0.9)
        else:
            # Reject step, reduce h
            h *= max(0.1, 0.9 * err ** (-0.25))
        
        # Deadline check
        if time.time() > _DEADLINE - 5.0:
            break
    
    return y


def _backward_euler(f, t0, t1, y0, steps):
    """
    Implicit Euler with Newton iteration (handles stiff problems).
    """
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    
    for _ in range(steps):
        z = y.copy()
        for _newton in range(10):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            try:
                step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                break
            z = z + step
            if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
        
        if time.time() > _DEADLINE - 5.0:
            break
    
    return y


def integrate(f, t_span, y0):
    """
    Main integration function. Tries linear system solver first, then falls back to adaptive RK45.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # First, try to solve as a linear time-invariant system
    result = _solve_linear_system(f, t_span, y0)
    if result is not None and np.all(np.isfinite(result)):
        return result
    
    # Fall back to adaptive RK45
    result = _rk45_adaptive(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=50000)
    if result is not None and np.all(np.isfinite(result)):
        return result
    
    # Final fallback: implicit Euler
    result = _backward_euler(f, t0, t1, y0, 2000)
    return result
# EVOLVE-BLOCK-END
