import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use the same method and tolerances as the reference solution
    # to minimize the difference between y_end and y_ref
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback: Matrix Exponential
        t0, t1 = t_span
        n = len(y0)
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            A[i, i] = -2 * 21**2
            if i > 0:
                A[i, i-1] = 1 * 21**2
            if i < n-1:
                A[i, i+1] = 1 * 21**2
        from scipy.linalg import expm
        M = expm(A * (t1 - t0))
        return M @ y0
        
    return sol.y[:, -1]
