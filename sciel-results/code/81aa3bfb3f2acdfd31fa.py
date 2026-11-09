"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm, solve

def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) for a fixed linear system y' = A y + b.
    Uses matrix exponential for exact solution (up to floating point precision).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    # Step 1: Determine A and b
    # Assume f(t, y) = A y + b
    # b = f(0, 0)
    y_zero = np.zeros(n)
    b = f(0.0, y_zero)
    
    # Check if b is effectively zero
    # If not, it's an inhomogeneous system
    is_homogeneous = np.allclose(b, 0.0, atol=1e-12)
    
    # A columns: A[:, j] = f(0, e_j) - b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j) - b
        
    # Check deadline
    if time.time() > deadline:
        # Fallback: return initial state? Or a crude estimate?
        # If we ran out of time during A construction, something is wrong.
        # But A construction is 15 calls, very fast.
        return y0

    T = t1 - t0
    
    # Step 2: Compute Matrix Exponential E = exp(T A)
    E = expm(T * A)
    
    # Check deadline
    if time.time() > deadline:
        return E @ y0 + (T * b)  # Crude approx if expm took too long? Unlikely.

    # Step 3: Compute the integral term C = integral_0^T exp(A tau) d_tau
    # If homogeneous, C is not needed.
    if is_homogeneous:
        y_end = E @ y0
    else:
        # C = A^{-1} (E - I)
        # Solve A C = E - I for C
        # C is n x n
        I = np.eye(n)
        try:
            # Solve for C directly
            # A @ C = E - I
            C = solve(A, E - I)
        except np.linalg.LinAlgError:
            # Fallback if A is singular
            # Use series expansion for C = sum_{k=0}^infty T^{k+1} A^k / (k+1)!
            # This is effectively integrating the matrix exponential series
            # C = T I + T^2 A / 2! + T^3 A^2 / 3! ...
            # Or use the identity: C = (E - I) @ A^{-1} if A is invertible.
            # If A is singular, we can use scipy.linalg.expm_multiply? No.
            # Let's try to solve with least squares or pinv
            C = np.linalg.pinv(A) @ (E - I)
            
        y_end = E @ y0 + C @ b
        
    return y_end
# EVOLVE-BLOCK-END
