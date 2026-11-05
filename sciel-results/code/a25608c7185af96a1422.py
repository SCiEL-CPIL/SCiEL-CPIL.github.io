"""
Rewritten program: ODE solver for 15-dim stiff linear system using matrix exponential.
Assumes f(t, y) = A @ y where A is a constant matrix.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float).flatten()
    n = len(y0)
    
    # If dimension is 0, return empty array
    if n == 0:
        return np.array([])
        
    # Check for affine term b = f(0, 0)
    # If b is non-zero, the system is y' = A y + b
    b = np.array(f(0.0, np.zeros(n, dtype=float)), dtype=float)
    is_affine = np.linalg.norm(b) > 1e-10
    
    # Construct the matrix A by evaluating f on the standard basis vectors.
    # If affine, f(0, e_j) = A e_j + b, so A[:, j] = f(0, e_j) - b
    # If homogeneous, f(0, e_j) = A e_j, so A[:, j] = f(0, e_j)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        if is_affine:
            A[:, j] -= b
            
    # Calculate the matrix exponential exp(A * (t1 - t0))
    # The solution to y' = A y is y(t) = exp(A * t) y(0)
    # The solution to y' = A y + b is y(t) = exp(A * t) y(0) + (exp(A * t) - I) A^{-1} b
    dt = t1 - t0
    if dt == 0:
        return y0.copy()
        
    M = scipy.linalg.expm(A * dt)
    
    # Apply the matrix exponential to the initial state
    y_end = M @ y0
    
    if is_affine:
        # Compute the particular solution for the affine term
        # x = A^{-1} b
        # Use least squares in case A is singular or ill-conditioned
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.linalg.pinv(A) @ b
            
        # Particular solution: (M - I) x
        y_end += (M - np.eye(n)) @ x
        
    return y_end
# EVOLVE-BLOCK-END
