"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute a linear combination of 10 cubic B-spline basis functions
    evaluated at points x in [-3, 3].
    
    This uses a clamped cubic B-spline representation with 10 control points.
    The basis functions are computed via the Cox-de Boor recursion.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n == 0:
        return np.array([])

    # Map x from [-3, 3] to [0, 1]
    u = (x + 3.0) / 6.0
    
    # Clamp u to [0, 1] to handle any floating point edge cases
    u = np.clip(u, 0.0, 1.0)

    # Define knots for clamped cubic B-splines with N=10 control points
    # Knot vector length M = N + p + 1 = 10 + 3 + 1 = 14
    # Clamped: 4 zeros, 6 internal, 4 ones
    # Internal knots evenly spaced: i/7 for i in 1..6
    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0/7.0, 2.0/7.0, 3.0/7.0, 4.0/7.0, 5.0/7.0, 6.0/7.0,
        1.0, 1.0, 1.0, 1.0
    ])
    
    p = 3
    n_ctrl = 10
    
    # Evaluate B-spline basis functions using Cox-de Boor recursion.
    # We compute a matrix B of shape (n_ctrl, n) where B[i, j] is the value 
    # of the i-th basis function at x_j.
    
    # Initialize N_i,0
    N_prev = np.zeros((n_ctrl, n))
    
    # p=0 step
    # N_i,0(u) = 1 if knots[i] <= u < knots[i+1]
    # Special case: for the last basis function, include the right endpoint u=1
    for i in range(n_ctrl):
        k_left = knots[i]
        k_right = knots[i+1]
        
        # Skip zero-width intervals (due to clamped knots)
        if k_left == k_right:
            continue
            
        if i == n_ctrl - 1:
            # Include right boundary for the last one to ensure partition of unity at u=1
            N_prev[i, :] = (u >= k_left) & (u <= k_right)
        else:
            N_prev[i, :] = (u >= k_left) & (u < k_right)
            
    # Recursion for p=1, 2, 3
    for degree in range(1, p + 1):
        N_curr = np.zeros((n_ctrl, n))
        for i in range(n_ctrl):
            # Term 1: (u - u_i) / (u_{i+p} - u_i) * N_i,p-1
            denom1 = knots[i + degree] - knots[i]
            if denom1 != 0:
                coeff1 = (u - knots[i]) / denom1
                N_curr[i, :] += coeff1 * N_prev[i, :]
            
            # Term 2: (u_{i+p+1} - u) / (u_{i+p+1} - u_{i+1}) * N_{i+1},p-1
            if i + 1 < n_ctrl:
                denom2 = knots[i + degree + 1] - knots[i + 1]
                if denom2 != 0:
                    coeff2 = (knots[i + degree + 1] - u) / denom2
                    N_curr[i, :] += coeff2 * N_prev[i + 1, :]
        
        N_prev = N_curr

    # N_prev is now the matrix of basis functions (10, n)
    # result = N_prev.T @ params
    
    # Check for numerical stability / deadline
    if time.time() > deadline:
         # Just return what we have so far, though it should be fast
         pass

    result = N_prev.T @ params

    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)

    return result
# EVOLVE-BLOCK-END
