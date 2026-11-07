"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute a linear combination of 10 cubic B-spline basis functions
    evaluated at points x in [-3, 3].
    """
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n == 0:
        return np.array([])

    # Parameters for the B-spline representation
    # We define 10 control points, which correspond to 7 internal knots for cubic splines
    # (degree=3, number_of_points=10 -> number_of_knots = 10 - 3 - 1 = 6 internal knots?
    # Let's check: For N control points and degree p, we need N+p+1 knots total.
    # If we want the spline to span [-3, 3], we can place knots evenly.
    # Let's use 10 control points.
    # Knots: -3, -2, -1, 0, 1, 2, 3 ... wait, that's 7 knots.
    # N=10, p=3. Total knots = N+p+1 = 14.
    # We need 14 knots.
    # Let's distribute them over a slightly larger domain to ensure the basis functions
    # are well-defined and non-degenerate, or just use clamped B-splines.
    
    # Simplest robust approach:
    # Map x to [0, 1] to make knot placement standard.
    # Let u = (x + 3) / 6.0, so u is in [0, 1].
    # We want 10 basis functions.
    # For a clamped cubic B-spline with 10 control points, we need:
    # Knot vector: 0, 0, 0, 0, 1/9, 2/9, ..., 8/9, 1, 1, 1, 1
    # This has 4 zeros, 7 internal, 4 ones -> Total 15 knots?
    # N=10, p=3. Knot vector length M = N + p + 1 = 14.
    # Clamped: 0, 0, 0, 0, ..., 1, 1, 1, 1
    # Number of internal knots = M - 2*(p+1) = 14 - 8 = 6.
    # So knots: [0, 0, 0, 0, k1, k2, k3, k4, k5, k6, 1, 1, 1, 1]
    # Let's choose k_i = i/7 for i in 1..6.
    
    u = (x + 3.0) / 6.0
    
    # Define knots
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0/7.0, 2.0/7.0, 3.0/7.0, 4.0/7.0, 5.0/7.0, 6.0/7.0, 1.0, 1.0, 1.0, 1.0])
    
    # We need to evaluate the 10 basis functions N_i,3(u) for i=0..9.
    # We can use the Cox-de Boor recursion.
    # N_i,0(u) = 1 if u_i <= u < u_{i+1} else 0
    # N_i,p(u) = (u - u_i) / (u_{i+p} - u_i) * N_i,p-1(u) + (u_{i+p+1} - u) / (u_{i+p+1} - u_{i+1}) * N_{i+1},p-1(u)
    
    p = 3
    n_ctrl = 10
    n_knots = len(knots)
    
    # Initialize N_i,0
    # Be careful with the rightmost knot for the last interval.
    # Usually, the last basis function is defined on [u_{n-1}, u_n] including the right endpoint.
    
    # We will compute the matrix of basis values for all x points at once for efficiency.
    # Shape: (n_ctrl, n)
    
    # Step 0: p=0
    # N_i,0 is 1 if u_i <= u < u_{i+1}
    # For i=0..9 (indices 0 to 9 in 0-based, corresponding to control points)
    # Knots indices for N_i,0 are i to i+1.
    # Note: The last interval is special. Let's handle it by extending the domain slightly or using <= for the last one.
    # For B-splines, the domain is [u_p, u_{n-p}]. Here u_3=0, u_{10}=1 (if knots are 0..1).
    # Wait, my knots are [0,0,0,0, ..., 1,1,1,1].
    # u_3 = 0, u_{n-knots} ... let's just iterate.
    
    N_prev = np.zeros((n_ctrl, n))
    
    # p=0
    for i in range(n_ctrl):
        # Interval [knots[i], knots[i+1])
        # Exception: for the last control point, we might want to include the right end if u==1
        lower = knots[i]
        upper = knots[i+1]
        
        if i == n_ctrl - 1:
            # Last basis function
            N_prev[i, :] = (u >= lower) & (u <= upper)
        else:
            N_prev[i, :] = (u >= lower) & (u < upper)
    
    # Handle the case where u == 1.0 exactly.
    # In standard B-splines, the last knot is the right boundary.
    # If u=1, only N_{n-1,0} is non-zero?
    # Actually, with clamped knots, if u=1, it falls in the last interval [1, 1)? No, that's empty.
    # We need to be careful. The definition usually is u_i <= u < u_{i+1}.
    # For the last point u=1, we should force the last basis function to be 1 (if it's the last one).
    # Let's re-evaluate p=0 for u=1.
    # If u=1, and knots are ..., 1, 1, 1, 1.
    # N_9,0 is defined on [knots[9], knots[10]) = [1/7 * 6? No, knots[9]=6/7, knots[10]=1.
    # Wait, my knots list:
    # 0:0, 1:0, 2:0, 3:0, 4:1/7, 5:2/7, 6:3/7, 7:4/7, 8:5/7, 9:6/7, 10:1, 11:1, 12:1, 13:1
    # N_0,0: [0,0) -> empty?
    # Actually, for clamped splines, the first few intervals are zero-width.
    # N_0,0 is 0 because 0 <= u < 0 is false.
    # This is correct. The non-zero basis functions at u=0 will emerge after recursion.
    
    # Let's refine the p=0 step.
    N_prev = np.zeros((n_ctrl, n))
    for i in range(n_ctrl):
        k_left = knots[i]
        k_right = knots[i+1]
        if k_left == k_right:
            continue
        if i == n_ctrl - 1:
            # Include right boundary for the last one to ensure partition of unity at u=1
            N_prev[i, :] = (u >= k_left) & (u <= k_right)
        else:
            N_prev[i, :] = (u >= k_left) & (u < k_right)
            
    # It is possible that for some x, no basis function is non-zero at p=0?
    # If u is in [0, 1], it should be covered.
    # However, if u < 0 or u > 1, they will be all zeros.
    # Since x in [-3, 3], u in [0, 1], this should be fine.
    
    # Recursion for p=1, 2, 3
    for degree in range(1, p + 1):
        N_curr = np.zeros((n_ctrl, n))
        for i in range(n_ctrl):
            # Term 1
            denom1 = knots[i + degree] - knots[i]
            if denom1 != 0:
                coeff1 = (u - knots[i]) / denom1
                N_curr[i, :] += coeff1 * N_prev[i, :]
            
            # Term 2
            # N_{i+1}, degree-1
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
