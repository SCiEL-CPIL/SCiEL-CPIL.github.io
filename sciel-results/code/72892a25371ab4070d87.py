"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n == 0:
        return np.array([])

    # Map x from [-3, 3] to [0, 1] for B-spline evaluation
    u = (x + 3.0) / 6.0

    # We want 10 cubic B-spline basis functions.
    # For clamped cubic splines with N=10 control points:
    # Knot vector length M = N + p + 1 = 10 + 3 + 1 = 14.
    # Clamped boundary conditions require 4 repeated knots at each end.
    # Internal knots: M - 2*(p+1) = 14 - 8 = 6.
    # Let's place internal knots evenly in (0, 1).
    # Knots: [0, 0, 0, 0, k1, k2, k3, k4, k5, k6, 1, 1, 1, 1]
    # k_i = i / 7.0 for i in 1..6
    
    knots = np.array([
        0.0, 0.0, 0.0, 0.0,
        1.0/7.0, 2.0/7.0, 3.0/7.0, 4.0/7.0, 5.0/7.0, 6.0/7.0,
        1.0, 1.0, 1.0, 1.0
    ])
    
    p = 3
    n_ctrl = 10
    n_knots = len(knots)
    
    # Evaluate B-spline basis functions using Cox-de Boor recursion.
    # We compute a matrix B of shape (n_ctrl, n) where B[i, j] is the value 
    # of the i-th basis function at x_j.
    
    # Initialize N_i,0
    N_prev = np.zeros((n_ctrl, n))
    
    # p=0 step
    # N_i,0(u) = 1 if knots[i] <= u < knots[i+1]
    # Special case for the last interval to include u=1.0
    for i in range(n_ctrl):
        k_left = knots[i]
        k_right = knots[i+1]
        
        if k_left == k_right:
            continue
            
        if i == n_ctrl - 1:
            # Include right boundary for the last basis function
            N_prev[i, :] = (u >= k_left) & (u <= k_right)
        else:
            N_prev[i, :] = (u >= k_left) & (u < k_right)
            
    # Handle edge case: if u is exactly 1.0, ensure the last basis function is active
    # The loop above handles this if knots[9] < 1.0 <= knots[10]
    # knots[9] = 6/7, knots[10] = 1.0. So for i=9, (u >= 6/7) & (u <= 1.0) is True for u=1.0.
    # For i=8, knots[8]=5/7, knots[9]=6/7. (u >= 5/7) & (u < 6/7) is False for u=1.0.
    # So only the last one is active at p=0 for u=1.0? 
    # Wait, standard B-spline definition for the last knot span often includes the right endpoint.
    # The recursion will handle the spread.
    
    # Recursion for p=1, 2, 3
    for degree in range(1, p + 1):
        N_curr = np.zeros((n_ctrl, n))
        for i in range(n_ctrl):
            # Term 1: (u - u_i) / (u_{i+p} - u_i) * N_i, p-1
            denom1 = knots[i + degree] - knots[i]
            if denom1 != 0:
                coeff1 = (u - knots[i]) / denom1
                N_curr[i, :] += coeff1 * N_prev[i, :]
            
            # Term 2: (u_{i+p+1} - u) / (u_{i+p+1} - u_{i+1}) * N_{i+1}, p-1
            if i + 1 < n_ctrl:
                denom2 = knots[i + degree + 1] - knots[i + 1]
                if denom2 != 0:
                    coeff2 = (knots[i + degree + 1] - u) / denom2
                    N_curr[i, :] += coeff2 * N_prev[i + 1, :]
        
        N_prev = N_curr

    # N_prev is now the basis matrix (10, n)
    # result = sum(params[i] * N_prev[i, :])
    result = N_prev.T @ params
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
