import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)

    # Cubic B-spline basis with 10 control points
    # Domain: [-3, 3]
    # Clamped knot vector for N=10, k=3:
    # t_0..t_3 = -3
    # t_10..t_13 = 3
    # Internal knots t_4..t_9 equally spaced
    
    knots = np.zeros(14)
    knots[0:4] = -3.0
    knots[10:14] = 3.0
    # Internal knots: 6 knots between -3 and 3
    # t_4, t_5, t_6, t_7, t_8, t_9
    # Step size = 6 / 6 = 1.0
    # t_4 = -2, t_5 = -1, t_6 = 0, t_7 = 1, t_8 = 2, t_9 = 3? No, t_9 must be < 3.
    # Actually, let's use linspace for the internal knots to ensure even spacing.
    # There are 6 internal knots.
    # Let's just place them at -2, -1, 0, 1, 2, and... wait, 6 knots.
    # -2, -1, 0, 1, 2 is only 5.
    # Let's use: -2, -1, 0, 1, 2, 2.5? No, uniform is better.
    # The internal knots should be in (-3, 3).
    # Let's use linspace(-3, 3, 8)[1:-1] which gives 6 points.
    # linspace(-3, 3, 8) = [-3, -2, -1, 0, 1, 2, 3] ... wait, 7 points.
    # linspace(-3, 3, 8) = [-3, -1.8, -0.6, 0.6, 1.8, 3]? No.
    # np.linspace(-3, 3, 8) has step 6/7.
    # Points: -3, -3+6/7, -3+12/7, -3+18/7, -3+24/7, -3+30/7, -3+36/7, 3.
    # Indices 1 to 6 are the internal ones.
    internal_knots = np.linspace(-3, 3, 8)[1:-1]
    knots[4:10] = internal_knots
    
    # Ensure knots are strictly increasing where required, or at least non-decreasing
    # For B-splines, repeated knots are allowed.
    # Let's verify the knot vector:
    # -3, -3, -3, -3, -2.14, -1.28, -0.43, 0.43, 1.28, 2.14, 3, 3, 3, 3
    # This looks valid.

    n_basis = 10
    k = 3
    
    # Initialize basis matrix B of shape (len(x), n_basis)
    B = np.zeros((n, n_basis))
    
    # Degree 0
    for i in range(n_basis):
        # N_i,0(x) = 1 if t_i <= x < t_{i+1} else 0
        # Handle the last interval closure
        if i < n_basis - 1:
            B[:, i] = np.where((x >= knots[i]) & (x < knots[i+1]), 1.0, 0.0)
        else:
            # Last basis function: include the right endpoint
            B[:, i] = np.where((x >= knots[i]) & (x <= knots[i+1]), 1.0, 0.0)
    
    # Degrees 1, 2, 3
    for d in range(1, k + 1):
        new_B = np.zeros_like(B)
        for i in range(n_basis - d):
            # Term 1
            denom1 = knots[i + d] - knots[i]
            if denom1 != 0:
                new_B[:, i] += (x - knots[i]) / denom1 * B[:, i]
            
            # Term 2
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 != 0:
                new_B[:, i] += (knots[i + d + 1] - x) / denom2 * B[:, i + 1]
        
        B = new_B
    
    # The B-spline basis functions should sum to 1 (partition of unity)
    # for x in [knots[k], knots[-k-1]] which is [knots[3], knots[10]] = [-3, 3].
    # So this should be well-behaved.
    
    # Return linear combination
    result = B @ params
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    return result
