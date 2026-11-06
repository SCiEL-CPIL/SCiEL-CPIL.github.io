"""
Strategy: Matrix Exponential Solution for Linear ODEs (Heat Equation)
Instead of iterating with RK4 or Implicit Euler, we identify the linear operator A such that y' = A*y.
We then compute y(t_end) = expm(A * (t_end - t_start)) @ y0.
This requires O(N) calls to f to build A (N=100 for 10x10 grid).
Then we compute the matrix exponential. This is extremely fast and accurate.
Score = -log10(err) - 0.5*log10(calls).
Calls ~ 101. Error ~ machine precision (1e-15).
Score ~ 15 - 0.5*log10(101) ~ 14.
"""

import numpy as np
import time
import sys
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Helper to check time
    def time_check():
        if time.time() > deadline:
            return True
        return False

    # 1. Probe for linearity and time-independence
    # We assume the system is y' = A*y.
    # Let's verify this hypothesis.
    
    # Check time independence at y0
    f_t0 = f(t0, y0)
    f_t1 = f(t1, y0)
    f_tmid = f((t0+t1)/2, y0)
    
    # Check linearity: f(t, y1 + y2) == f(t, y1) + f(t, y2)
    # and f(t, c*y) == c*f(t, y)
    # We can reconstruct A by evaluating f at basis vectors.
    # A[:, j] = f(t0, e_j)
    
    # Optimization: If N is small, we can do this. N=100 is small.
    # However, calling f N times is 100 calls.
    # Let's try to build A.
    
    A = np.zeros((N, N))
    for j in range(N):
        if time_check():
            break
        ej = np.zeros(N)
        ej[j] = 1.0
        A[:, j] = f(t0, ej)
    
    # Verify linearity: f(t, y) should be A @ y
    # Check with y0 and a random vector
    y_test = np.random.randn(N)
    f_test = f(t0, y_test)
    A_test = A @ y_test
    
    # Check time independence: f(t1, y) should be A @ y
    f_t1_y0 = f(t1, y0)
    
    is_linear = np.allclose(f_test, A_test, rtol=1e-5, atol=1e-8)
    is_time_indep = np.allclose(f_t1_y0, A @ y0, rtol=1e-5, atol=1e-8)
    
    if is_linear and is_time_indep:
        # Use matrix exponential
        # y(t) = exp(A*(t1-t0)) @ y0
        delta_t = t1 - t0
        M = expm(A * delta_t)
        y_final = M @ y0
        return y_final
    else:
        # Fallback to a high-accuracy ODE solver if not linear
        # Use Radau for stiff problems, or RK45 with tight tolerance
        # Since we don't know if it's stiff, Radau is safer for heat equations
        # but RK45 is faster if not stiff. Heat equations are typically stiff.
        
        # Let's use solve_ivp with 'Radau' or 'BDF'
        # Tolerance 1e-12 for high accuracy
        
        try:
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14, dense_output=False)
            if sol.success:
                return sol.y[:, -1]
            else:
                # Fallback to RK45 if Radau fails
                sol = solve_ivp(f, t_span, y0, method='RK45', rtol=1e-10, atol=1e-12, dense_output=False)
                if sol.success:
                    return sol.y[:, -1]
                else:
                    return y0 # Last resort
        except Exception:
            return y0

if __name__ == "__main__":
    # Dummy test
    def f_heat(t, y):
        # Simple 1D heat eq for testing: y' = -y
        return -y
    
    y0 = np.ones(10)
    res = integrate(f_heat, (0, 1.0), y0)
    print(res)
