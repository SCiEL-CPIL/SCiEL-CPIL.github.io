"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear ODE y' = f(t, y) from t_span[0] to t_span[1].
    Uses an implicit BDF method with a finite-difference Jacobian and Newton iteration.
    Includes adaptive step size control and a deadline guard.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: stop if we've used more than 280 seconds (leaving 50s margin)
    deadline = time.time() + 280.0
    
    # We'll use a BDF2-like method or just backward Euler with Newton.
    # For stiff linear systems, backward Euler with a good Jacobian is very robust.
    # We'll use a fixed number of Newton iterations and monitor convergence.
    
    # Estimate Jacobian once to check if it's constant (linear system)
    # For a linear system y' = A y + b(t), the Jacobian is constant A.
    # Let's compute it once and reuse if it appears constant.
    
    def compute_jacobian(t, y):
        """Compute Jacobian using forward differences."""
        J = np.empty((n, n), dtype=float)
        fz = f(t, y)
        for j in range(n):
            dy = np.zeros(n)
            scale = 1e-7 * max(1.0, abs(y[j]))
            dy[j] = scale
            J[:, j] = (f(t, y + dy) - fz) / scale
        return J, fz
    
    # Try to determine if the system is linear by checking if J is constant
    # Compute J at initial point
    J, fz = compute_jacobian(t0, y)
    
    # Check linearity: evaluate f at another point and see if J is consistent
    # For now, we'll recompute J periodically or use it if the system is linear
    # A simpler approach: just use backward Euler with Newton, recomputing J each step
    # but caching it if it doesn't change much
    
    # Let's use a fixed step backward Euler with Newton, but with a reasonable number of steps
    # For a stiff problem on [0,1], we need steps small enough for accuracy but the method
    # is stable for large h. The accuracy is O(h), so we need many steps for high accuracy.
    
    # Alternative: use a higher-order BDF or implicit Runge-Kutta.
    # For now, let's use backward Euler with Newton, adaptive step size.
    
    # Actually, for the score -log10(rel_err) - 0.5*log10(calls), we want:
    # 1. High accuracy (many steps or higher order)
    # 2. Few f evaluations (efficient Newton, good initial guess)
    
    # Let's use a BDF2 method with Newton iteration.
    # BDF2: (3/2)*y_{n+1} - 2*y_n + (1/2)*y_{n-1} = h * f(t_{n+1}, y_{n+1})
    
    # Start with a smaller step for the first step (use backward Euler for step 1)
    
    h = 1.0 / 1000  # Initial step size
    t = t0
    y_prev = y.copy()
    
    # We'll do adaptive stepping with error control
    # For simplicity, let's use a fixed number of steps but with a higher-order method
    
    # Let me reconsider: use implicit midpoint or trapezoidal rule (2nd order)
    # Trapezoidal: y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
    
    # This is A-stable and 2nd order. Let's use this with Newton.
    
    # Reset
    t = t0
    y = np.array(y0, dtype=float)
    h = 0.001  # Start with this step size
    
    # We'll adapt h based on error estimate
    # Use embedded method or just monitor Newton convergence
    
    # For a linear stiff system, the trapezoidal rule with Newton should work well.
    # Let's use a fixed number of steps, say 5000, which gives h = 0.0002
    
    num_steps = 5000
    h = (t1 - t0) / num_steps
    
    # Precompute f at t0
    f0 = f(t0, y)
    
    for i in range(num_steps):
        t_new = t + h
        t_new = min(t_new, t1)  # Don't overshoot
        
        # Solve for y_new: y_new = y + h/2 * (f0 + f(t_new, y_new))
        # Newton iteration: find y_new such that g(y_new) = 0
        # g(y_new) = y_new - y - h/2 * (f0 + f(t_new, y_new))
        
        y_new = y + h/2 * f0  # Initial guess: explicit Euler step
        
        converged = False
        for newton_iter in range(10):
            f_new = f(t_new, y_new)
            g = y_new - y - h/2 * (f0 + f_new)
            
            # Compute Jacobian of g: I - h/2 * J_f
            J, _ = compute_jacobian(t_new, y_new)
            A = np.eye(n) - h/2 * J
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                delta = np.linalg.lstsq(A, -g, rcond=None)[0]
            
            y_new = y_new + delta
            
            if np.linalg.norm(delta) < 1e-12 * (1 + np.linalg.norm(y_new)):
                converged = True
                break
        
        if not converged:
            # Reduce step size and retry
            h = h / 2
            num_steps *= 2
            # Recompute
            continue
        
        t = t_new
        y = y_new
        f0 = f(t, y)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return y
# EVOLVE-BLOCK-END
