import numpy as np
import time
import sys

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin for 330s limit
    
    def time_check():
        return time.time() < deadline

    # 1. Probe for Linearity and Time-Invariance
    # The heat equation is linear and autonomous: y' = A * y
    # We verify this by checking if f(t, y) = J * y for a few points.
    
    # Compute Jacobian J at (t0, y0) using central differences for accuracy
    # Central difference: (f(y+h) - f(y-h)) / 2h
    # This requires 2*n calls.
    # However, for 10x10=100, this is 200 calls. Very cheap.
    
    eps = 1e-7
    J = np.zeros((n, n))
    
    # First, get f(t0, y0)
    try:
        f0 = f(t0, y0_arr)
    except Exception:
        return _fallback(f, t_span, y0_arr)

    # Check if f is linear and autonomous
    # We need to verify f(t, y) = J y.
    # Let's estimate J using finite differences at y0.
    for j in range(n):
        if not time_check():
            break
        e = np.zeros(n)
        e[j] = eps * max(1.0, abs(y0_arr[j]))
        
        y_plus = y0_arr + e
        y_minus = y0_arr - e
        
        try:
            f_plus = f(t0, y_plus)
            f_minus = f(t0, y_minus)
        except Exception:
            return _fallback(f, t_span, y0_arr)
            
        J[:, j] = (f_plus - f_minus) / (2.0 * e[j])
        
    if not np.all(np.isfinite(J)):
        return _fallback(f, t_span, y0_arr)

    # Now, verify that f(t, y) = J * y for a few random points
    # This ensures the operator is constant (autonomous) and linear.
    # Pick 3 random test points.
    np.random.seed(42)
    is_linear = True
    for _ in range(3):
        if not time_check():
            break
        y_test = np.random.randn(n) * 0.1 + 0.5 # Some non-trivial values
        t_test = t0 + 0.01 # Slightly different time
        
        try:
            f_test = f(t_test, y_test)
        except Exception:
            is_linear = False
            break
            
        Jy_test = J.dot(y_test)
        
        # Check relative error
        norm_diff = np.linalg.norm(f_test - Jy_test)
        norm_ref = np.linalg.norm(Jy_test) + 1e-10
        if norm_diff / norm_ref > 1e-5:
            is_linear = False
            break
            
    if is_linear:
        # We have confirmed y' = J y.
        # Solution: y(t1) = expm((t1 - t0) * J) @ y0
        # Use scipy.linalg.expm for stability.
        try:
            from scipy.linalg import expm
            dt = t1 - t0
            M = dt * J
            expM = expm(M)
            y_final = expM.dot(y0_arr)
            return y_final
        except Exception:
            pass # Fall through to fallback if expm fails

    # Fallback: If linearity check failed or expm failed, use a robust solver.
    # Since we have used ~200+ calls, we have budget for a decent ODE solver.
    return _fallback(f, t_span, y0_arr)

def _fallback(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Use RK45 or similar. 
    # Given the heat equation is stiff, explicit RK4 might need small steps.
    # Implicit methods are better, but let's stick to a robust explicit with many steps
    # or a simple implicit Euler if we can't use SciPy ODEs easily.
    # Actually, let's just use a high-step RK4.
    # Heat equation stability for RK4: h < 2.8 * dx^2 / (4 * D) ... depends on grid.
    # For 10x10, dx=1/10? Or dx=1?
    # If dx=1, stability limit is ~0.7. t_span is 0.05. So N=100 is likely stable.
    # If dx=0.1, stability limit is ~0.007. N=1000 is needed.
    # Let's try N=1000 RK4. 4000 calls.
    
    N = 1000
    h = (t1 - t0) / N
    t = t0
    y = y0.copy()
    
    for _ in range(N):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
