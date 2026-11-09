"""ODE solver: 15-dimensional stiff linear system using implicit BDF methods."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    
    Uses an implicit BDF (Backward Differentiation Formula) method with
    Newton iteration and finite-difference Jacobian, specifically designed
    for stiff systems. Falls back to Radau if BDF struggles.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Count function calls
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # First, try to determine if the system is linear by checking f(t, y) structure
    # For a linear system y' = A(t)*y + b(t), we can potentially solve more efficiently
    # But let's use a robust implicit method first
    
    # Use BDF method with adaptive stepping
    # Start with a reasonable initial step size
    t_total = t1 - t0
    
    # Try BDF with strict tolerances
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01,
            first_step=1e-4,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If BDF failed, try Radau
    call_count[0] = 0
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01,
            first_step=1e-4,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If both failed, try a custom implicit Euler with more Newton iterations
    call_count[0] = 0
    y = y0.copy()
    t = t0
    
    # Adaptive step size implicit Euler
    h = 1e-4
    while t < t1:
        h = min(h, t1 - t)
        
        # Newton iteration for implicit Euler: y_new = y + h * f(t + h, y_new)
        y_new = y.copy()
        
        for newton_iter in range(50):
            fz = f_counted(t + h, y_new)
            
            # Finite difference Jacobian
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1e-8, 1e-8 * abs(y_new[j]))
                e[j] = scale
                J[:, j] = (f_counted(t + h, y_new + e) - fz) / scale
            
            # Solve: (I - h*J) * delta = -(y_new - y - h*fz)
            A = np.eye(n) - h * J
            rhs = -(y_new - y - h * fz)
            
            try:
                delta = solve(A, rhs, assume_a='gen')
            except Exception:
                # Fallback to least squares
                delta, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            
            y_new = y_new + delta
            
            if np.linalg.norm(delta) < 1e-12 * (1 + np.linalg.norm(y_new)):
                break
        
        y = y_new
        t += h
        
        # Adaptive step control
        if newton_iter < 5:
            h = min(h * 2, 0.1)
        elif newton_iter > 20:
            h = max(h * 0.5, 1e-6)
        
        if not np.all(np.isfinite(y)):
            # Reset step and try smaller
            h = max(h * 0.1, 1e-8)
            y = y0 if t == t0 else y
            t = t - h if t > t0 else t0
    
    return y
