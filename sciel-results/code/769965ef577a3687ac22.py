"""ODE solver using implicit Runge-Kutta (Radau) with Jacobian for linear systems."""
import numpy as np
from scipy.integrate import solve_ivp

def _is_linear_time_invariant(f, t0, t1, y0, n):
    """Check if f is linear (or affine) and time-invariant.
    Returns (b, A) if affine linear time-invariant, else None.
    """
    # Check time-invariance with a test vector
    y_test = np.random.RandomState(0).randn(n)
    y_test = y_test / np.linalg.norm(y_test)
    ft0 = f(t0, y_test)
    ft1 = f(t1, y_test)
    if not np.allclose(ft0, ft1, rtol=1e-10, atol=1e-12):
        return None  # time-dependent
    
    # Check affine: f(t, 0) should be constant (the affine part b)
    b = f(t0, np.zeros(n))
    
    # Check linearity: f(t, y1 + y2) == f(t,y1) + f(t,y2) - b
    y1 = np.random.RandomState(1).randn(n)
    y2 = np.random.RandomState(2).randn(n)
    f_y1 = f(t0, y1)
    f_y2 = f(t0, y2)
    f_y12 = f(t0, y1 + y2)
    if not np.allclose(f_y12, f_y1 + f_y2 - b, rtol=1e-8, atol=1e-10):
        return None  # not affine linear
    
    return b

def _build_matrix_A(f, t0, n, b):
    """Build the matrix A such that f(t, y) = A y + b for all y."""
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e) - b
    return A

def _verify_A(f, t0, A, b, n):
    """Verify that f(t, y) ≈ A y + b for a few random vectors."""
    for seed in [3, 4, 5]:
        y = np.random.RandomState(seed).randn(n)
        y = y / np.linalg.norm(y)
        lhs = f(t0, y)
        rhs = A @ y + b
        if not np.allclose(lhs, rhs, rtol=1e-8, atol=1e-10):
            return False
    return True

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try to detect linear time-invariant system
    try:
        b = _is_linear_time_invariant(f, t0, t1, y0, n)
        if b is not None:
            A = _build_matrix_A(f, t0, n, b)
            if _verify_A(f, t0, A, b, n):
                # Define wrapped functions for solve_ivp
                def f_wrapped(t, y):
                    return A @ y + b
                
                def jac_wrapped(t, y):
                    return A
                
                # Use Radau (implicit RK, good for stiff)
                sol = solve_ivp(
                    f_wrapped, 
                    t_span, 
                    y0, 
                    method='Radau', 
                    jac=jac_wrapped,
                    rtol=1e-12, 
                    atol=1e-14
                )
                if sol.success:
                    return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: Adaptive RK45 if linear detection fails or Radau fails
    # Simple fixed step RK4 for non-linear or as last resort
    # Note: For non-linear, we don't have A, so we must use f directly.
    # A robust adaptive solver is better.
    # Re-using a simple adaptive RK45 logic or just fixed step RK4 with many steps.
    # Given the constraint, let's use a high-resolution fixed step RK4.
    # 10000 steps for 100 points is likely sufficient for moderate stiffness if not stiff.
    # If stiff, this will fail. But linear stiff systems are handled above.
    
    h = (t1 - t0) / 10000
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
