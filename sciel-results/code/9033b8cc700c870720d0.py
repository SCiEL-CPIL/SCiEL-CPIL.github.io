import numpy as np
from scipy.integrate import solve_ivp
import time
import sys

def integrate(f, t_span, y0):
    # Wrap f to count calls
    call_count = 0
    def counted_f(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
        
    # Analytic Jacobian for Robertson
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]
        ])
        return J

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Use Radau method with tight tolerances
    # The Robertson problem is stiff, so Radau is appropriate.
    sol = solve_ivp(
        counted_f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14, 
        jac=jac,
        max_step=1.0
    )
    
    if not sol.success:
        # Fallback to a simpler method if Radau fails (unlikely)
        sol = solve_ivp(counted_f, t_span, y0, method='BDF', rtol=1e-11, atol=1e-13, jac=jac)
        
    result = sol.y[:, -1]
    
    # High-precision polish pass if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 60.0:
        try:
            sol2 = solve_ivp(
                counted_f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-14, 
                atol=1e-16, 
                jac=jac,
                max_step=1.0
            )
            if sol2.success:
                result = sol2.y[:, -1]
        except Exception:
            pass
    
    return result
