"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time

def _rk4(f, t0, t1, y0, steps=2000):
    """
    Explicit 4th-order Runge-Kutta method.
    Used as a fallback if the system is not linear or if linear detection fails.
    """
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y

def _detect_linear(f, t0, y0):
    """
    Attempts to determine if y' = f(t, y) is of the form A*y + b.
    Returns (A, b) if linear, else (None, None).
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Get b = f(t0, 0) assuming we can call with 0? 
    # Better: b = f(t0, y0) - A*y0. But we don't have A yet.
    # Strategy: 
    # 1. Evaluate at y0 to get y0_dot.
    # 2. Evaluate at y0 + e_j * h to get columns of A.
    # 3. Check consistency.
    
    y0_dot = f(t0, y0)
    
    # To find A, we need to isolate the linear part.
    # If the system is strictly linear y' = A y, then f(t, 0) = 0.
    # Let's probe f(t0, 0) to check for affine offset b.
    # However, calling f(t, 0) might be expensive or unnecessary if we assume homogeneous.
    # The heat equation is typically homogeneous (y'=Ay).
    # Let's assume y' = A y first. If residual is high, try affine.
    
    A = np.zeros((n, n))
    h = 1e-6
    
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        y_perturbed = y0 + e
        y_perturbed_dot = f(t0, y_perturbed)
        
        # If y' = A y, then (y0 + e)_dot = A(y0 + e) = A y0 + A e
        # y0_dot = A y0
        # So A e = y_perturbed_dot - y0_dot
        # A e = A * [0..h..0] = h * A[:, j]
        # A[:, j] = (y_perturbed_dot - y0_dot) / h
        
        A[:, j] = (y_perturbed_dot - y0_dot) / h
        
    # Verify linearity: check if f(t, y) approx A y for a random vector
    # Or check if f(t, y0) approx A y0
    # Residual = || f(t0, y0) - A y0 ||
    residual = np.linalg.norm(y0_dot - A @ y0)
    
    # Also check if f(t, 0) is close to 0 (homogeneous)
    # If not homogeneous, it's affine y' = A y + b.
    # b = f(t, 0).
    # Let's check f(t0, 0).
    try:
        f_zero = f(t0, np.zeros(n))
        b = f_zero
        # Re-verify with affine model: y' = A y + b
        # A should be the same.
        # Residual check: || y0_dot - (A y0 + b) ||
        residual_affine = np.linalg.norm(y0_dot - (A @ y0 + b))
        
        if residual_affine < 1e-5 * (1 + np.linalg.norm(y0_dot)):
            return A, b
        else:
            # Try homogeneous again if affine failed? 
            # Or just return None if both are bad.
            if residual < 1e-5 * (1 + np.linalg.norm(y0_dot)):
                return A, np.zeros(n)
            else:
                return None, None
    except Exception:
        # If f(t, 0) fails or is not allowed, assume homogeneous
        if residual < 1e-5 * (1 + np.linalg.norm(y0_dot)):
            return A, np.zeros(n)
        else:
            return None, None

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280.0  # 20s margin
    
    # Try linear detection
    A, b = _detect_linear(f, t0, y0)
    
    if A is not None:
        # Check time
        if time.time() > deadline:
            # Fallback immediately
            return _rk4(f, t0, t1, y0, steps=500)
        
        try:
            dt = t1 - t0
            # Compute matrix exponential
            # expm(dt * A)
            M = sla.expm(dt * A)
            
            # Solution: y(t) = e^{At} y0 + (e^{At} - I) A^{-1} b  (if A invertible)
            # Or more generally for affine y' = Ay + b:
            # y(t) = e^{At} y0 + \int_0^t e^{A(t-\tau)} b d\tau
            # If b is constant, integral is (e^{At} - I) A^{-1} b
            # If A is singular, use series or solve (I - e^{At}) x = ...
            # For heat equation, A is usually negative definite, so invertible.
            # However, if b is zero, we just do M @ y0.
            
            if np.allclose(b, 0):
                y_end = M @ y0
            else:
                # Solve (I - M) x = -b ? No.
                # \int_0^t e^{A \tau} d\tau b = (e^{At} - I) A^{-1} b
                # Let K = e^{At} - I. We want K A^{-1} b.
                # This is equivalent to solving A X = K b ? No.
                # (e^{At} - I) A^{-1} b = v.
                # A v = (e^{At} - I) b.
                # So solve A v = (M - I) @ b.
                try:
                    rhs = (M - np.eye(n)) @ b
                    v = np.linalg.solve(A, rhs)
                    y_end = M @ y0 + v
                except np.linalg.LinAlgError:
                    # Fallback to numerical integration if A is singular
                    # Use RK4 with small steps
                    y_end = _rk4(f, t0, t1, y0, steps=1000)
            
            # Sanity check
            if not np.all(np.isfinite(y_end)):
                y_end = _rk4(f, t0, t1, y0, steps=1000)
                
            return y_end
            
        except Exception:
            # Fallback if expm fails
            return _rk4(f, t0, t1, y0, steps=1000)
    else:
        # Not linear, use RK4
        # Adaptive steps? For heat equation, stability might be an issue with large h.
        # 2000 steps for 0.05s is h=2.5e-5. Very small.
        # Let's use 1000 steps to be faster, error should still be small.
        return _rk4(f, t0, t1, y0, steps=1000)

# EVOLVE-BLOCK-END
