import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, leave 10s margin

    # 1. Probe for linearity
    # Check if f(t, y) = A y + b
    # Evaluate at t=0, y=0
    f_00 = f(t0, np.zeros(N))
    
    # Evaluate at t=0, y=y0
    f_0y0 = f(t0, y0)
    
    # Estimate Jacobian A at t=0
    A = np.zeros((N, N))
    delta = 1e-7
    for j in range(N):
        y_p = y0.copy()
        y_p[j] += delta
        f_p = f(t0, y_p)
        A[:, j] = (f_p - f_0y0) / delta
        
    # Check if A is constant (time independent)
    # Evaluate A at t=1
    A_1 = np.zeros((N, N))
    y_test = y0.copy() # Use y0 for consistency, or random
    f_1y0 = f(t1, y_test)
    for j in range(N):
        y_p = y_test.copy()
        y_p[j] += delta
        f_p = f(t1, y_p)
        A_1[:, j] = (f_p - f_1y0) / delta
        
    # Check linearity at a random point
    t_rand = (t0 + t1) / 2
    y_rand = np.random.randn(N) * np.linalg.norm(y0)
    f_rand = f(t_rand, y_rand)
    f_pred = A @ y_rand + f_00
    
    err = np.linalg.norm(f_rand - f_pred)
    scale = np.linalg.norm(A) * np.linalg.norm(y_rand) + np.linalg.norm(f_00) + 1e-12
    
    is_linear_constant = (err < 1e-5 * scale) and (np.linalg.norm(A - A_1) < 1e-5 * np.linalg.norm(A))
    
    if is_linear_constant:
        # Use Matrix Exponential
        # System: y' = A y + b
        # Augmented system: Y' = M Y, where Y = [y; 1], M = [[A, b], [0, 0]]
        M = np.zeros((N+1, N+1))
        M[:N, :N] = A
        M[:N, N] = f_00
        
        h = (t1 - t0)
        # One big step or multiple?
        # One step is exact for linear constant coeff.
        # But expm might be less stable for very large h? N=15 is small.
        # Let's use 1 step for minimum calls (0 calls for integration).
        E = scipy.linalg.expm(M * h)
        
        Y0 = np.zeros(N+1)
        Y0[:N] = y0
        Y0[N] = 1.0
        
        Y1 = E @ Y0
        return Y1[:N]
    else:
        # Fallback: solve_ivp with Radau (stiff)
        # This will use many f calls, but is robust.
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0, method='Radau', 
            rtol=1e-10, atol=1e-12,
            max_step=0.01 # Force some steps to ensure accuracy
        )
        return sol.y[:, -1]
