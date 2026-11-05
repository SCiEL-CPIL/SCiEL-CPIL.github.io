"""Solve a fixed 15-dimensional stiff linear IVP using matrix exponential methods."""
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    This solver assumes the ODE is a linear system y' = A(t) * y + b(t) where A and b
    are constant or can be determined from f. It uses matrix exponential methods
    for stability with stiff systems.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # First, determine if this is a linear system by probing f
    # For a linear system: f(t, y) = A @ y + b
    # We can determine A and b by evaluating f at multiple points
    
    # Evaluate at y0 to get b (approximately, if we assume the system is homogeneous or we can isolate b)
    f0 = np.asarray(f(t0, y0), dtype=float)
    
    # Probe to determine A: evaluate at y0 + e_i for each basis vector
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    
    # First, try to determine if the system is time-invariant
    # Evaluate at t0 and t1 with the same y to check
    f0_check = np.asarray(f(t1, y0), dtype=float)
    
    # Determine b by evaluating at y = 0
    y_zero = np.zeros(n, dtype=float)
    f_zero = np.asarray(f(t0, y_zero), dtype=float)
    b = f_zero.copy()
    
    # Determine A columns by evaluating at y = e_j
    for j in range(n):
        y_probe = np.zeros(n, dtype=float)
        y_probe[j] = 1.0
        f_probe = np.asarray(f(t0, y_probe), dtype=float)
        A[:, j] = f_probe - b
    
    # Check if A is constant by verifying at another time point
    A_check = np.zeros((n, n), dtype=float)
    for j in range(n):
        y_probe = np.zeros(n, dtype=float)
        y_probe[j] = 1.0
        f_probe = np.asarray(f(t1, y_probe), dtype=float)
        A_check[:, j] = f_probe - b
    
    # If A is approximately constant, use the closed-form solution with matrix exponential
    if np.allclose(A, A_check, atol=1e-10, rtol=1e-10):
        # Time-invariant linear system: y(t) = expm(A*(t-t0)) @ y0 + integral of expm(A*(t-tau)) @ b dtau
        # For constant b: y(t) = expm(A*(t-t0)) @ y0 + (expm(A*(t-t0)) - I) @ inv(A) @ b (if A is invertible)
        
        dt = t1 - t0
        expm_A = expm(A * dt)
        
        # Handle the particular solution for constant b
        # If A is invertible: particular = (expm(A*dt) - I) @ inv(A) @ b
        # If A is singular or near-singular, use a different approach
        try:
            A_inv = np.linalg.inv(A)
            particular = (expm_A - np.eye(n)) @ A_inv @ b
        except np.linalg.LinAlgError:
            # A is singular, use numerical integration for the particular solution
            # y_p(t) = integral_0^t expm(A*(t-tau)) @ b dtau
            # Use Simpson's rule or just approximate
            steps = 100
            h = dt / steps
            y_p = np.zeros(n, dtype=float)
            for i in range(steps):
                t_mid = i * h + h / 2
                y_p += expm(A * (dt - t_mid)) @ b * h
            particular = y_p
        
        y_final = expm_A @ y0 + particular
    else:
        # Time-varying case: use product of matrix exponentials with small steps
        # This is more complex; fall back to a robust implicit method
        # For now, use a fine-grained piecewise approach
        steps = 100
        h = (t1 - t0) / steps
        y = y0.copy()
        t = t0
        
        for i in range(steps):
            t_mid = t + h / 2
            # Determine A at this time
            A_local = np.zeros((n, n), dtype=float)
            b_local = np.asarray(f(t_mid, np.zeros(n, dtype=float)), dtype=float)
            
            for j in range(n):
                y_probe = np.zeros(n, dtype=float)
                y_probe[j] = 1.0
                f_probe = np.asarray(f(t_mid, y_probe), dtype=float)
                A_local[:, j] = f_probe - b_local
            
            # Use matrix exponential for this step
            expm_A_local = expm(A_local * h)
            try:
                A_inv_local = np.linalg.inv(A_local)
                particular_local = (expm_A_local - np.eye(n)) @ A_inv_local @ b_local
            except np.linalg.LinAlgError:
                particular_local = expm_A_local @ b_local * h  # Approximation
            
            y = expm_A_local @ y + particular_local
            t += h
        
        y_final = y
    
    return y_final
