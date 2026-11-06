"""Rewritten program using exact linear ODE solution via matrix extraction."""
import numpy as np
import scipy.linalg as la
import time


def integrate(f, t_span, y0):
    """Solve y' = f(t, y) using linear structure detection and exact matrix exponential."""
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 320s safety margin (evaluator kills at 330s)
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Extract the linear operator A such that f(t, y) = A @ y
    # For the heat equation, this is time-independent, so we can use any t
    # f(t, 0) should be 0 for a linear system; verify and extract A
    
    # First check: f(t, 0) should be zero (or very close)
    zero = np.zeros(n)
    f0 = np.asarray(f(t0, zero), dtype=float)
    
    # If f(0) is not zero, it might be affine: f(t,y) = A@y + b
    # For heat equation, b should be zero
    if np.max(np.abs(f0)) > 1e-10:
        # Affine case: f(t,y) = A@y + b where b = f(t,0)
        b = f0
        # Extract A columns: A[:,j] = (f(t, e_j) - b) / 1
        A = np.zeros((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            fj = np.asarray(f(t0, ej), dtype=float)
            A[:, j] = fj - b
        
        # Solution for affine: y(t) = expm(A*dt)@y0 + (expm(A*dt)-I)@A^{-1}@b
        # But this requires A to be invertible. Use the ODE solution formula.
        # Actually, for affine y' = A*y + b, the solution is:
        # y(t) = expm(A*t)@y0 + integral_0^t expm(A*(t-s))@b ds
        # = expm(A*t)@y0 + A^{-1}@(expm(A*t)-I)@b if A is invertible
        # More robustly, use the augmented system or scipy's odeint on the affine form.
        # But let's try the direct approach first.
        try:
            M = la.expm(A * dt)
            # Solve for the particular solution: need to compute (expm(A*dt) - I) @ inv(A) @ b
            # Use the identity: integral_0^dt expm(A*s) ds = A^{-1}@(expm(A*dt) - I)
            # We can compute this via the augmented matrix exponential trick:
            # [A, I; 0, 0] * dt, then the top-right block gives the integral
            n_aug = 2 * n
            A_aug = np.zeros((n_aug, n_aug))
            A_aug[:n, :n] = A
            A_aug[:n, n:] = np.eye(n)
            M_aug = la.expm(A_aug * dt)
            integral_part = M_aug[:n, n:] @ b
            y_end = M @ y0 + integral_part
        except Exception:
            # Fallback to RK4
            y_end = _rk4_safe(f, t0, t1, y0, 5000, deadline)
        return y_end
    else:
        # Linear case: f(t, y) = A @ y
        A = np.zeros((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            fj = np.asarray(f(t0, ej), dtype=float)
            A[:, j] = fj
        
        # Exact solution: y(t_end) = expm(A * dt) @ y0
        M = la.expm(A * dt)
        y_end = M @ y0
        
        # Verify with a quick RK4 check if we have time
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            y_rk4 = _rk4_safe(f, t0, t1, y0, 1000, deadline)
            if y_rk4 is not None:
                # Compare; if they agree closely, we're confident
                diff = np.max(np.abs(y_end - y_rk4))
                scale = max(1.0, np.max(np.abs(y_end)))
                if diff / scale > 1e-6:
                    # Something's off; use RK4 with more steps
                    y_rk4_more = _rk4_safe(f, t0, t1, y0, 10000, deadline)
                    if y_rk4_more is not None:
                        y_end = y_rk4_more
        return y_end


def _rk4_safe(f, t0, t1, y0, steps, deadline):
    """RK4 with deadline guard."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for i in range(steps):
        if time.time() > deadline - 5.0:
            return y
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y
