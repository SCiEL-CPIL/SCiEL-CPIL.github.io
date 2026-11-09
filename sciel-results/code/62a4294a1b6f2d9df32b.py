"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Global piecewise linear "hat function" basis with fixed knots across [-2, 2].
    # The model is strictly linear in params, so BFGS will converge to the global
    # least-squares solution quickly.
    #
    # We use 10 parameters. We construct a basis of 10 hat functions (or similar
    # linear pieces) that span the domain.
    # Let's define knots evenly spaced.
    # To use exactly 10 params, let's create 10 basis functions.
    # A common approach: use B-splines or simple hat functions.
    # Simple hat functions: each is 0 outside a local interval, 1 at a center, 0 at neighbors.
    # However, hat functions don't cover the boundaries well if we just do interior ones.
    # Let's use a set of 10 linear pieces or a combination.
    
    # Alternative: Use 10 "ramp" functions or just 10 evenly spaced hat functions.
    # Let's try 10 hat functions centered at evenly spaced points in [-2, 2].
    # Centers: -2, -1.5, -1, ..., 2? That's 9 points.
    # Let's use 10 centers: np.linspace(-2, 2, 10).
    # But hat functions need width.
    
    # A more robust linear basis for regression:
    # We can use a set of 10 basis functions that are linear combinations of x and x^2? No, that's not "piecewise linear".
    # The prompt suggests "global piecewise linear 'hat function' basis".
    
    # Let's define 10 hat functions.
    # Knots: Let's place 10 centers.
    # centers = np.linspace(-2, 2, 10)
    # width = 4.0 / 9.0 (distance between centers)
    # Hat function i is:
    #   0 if x < centers[i] - width
    #   (x - (centers[i] - width)) / width if centers[i] - width <= x < centers[i]
    #   ((centers[i] + width) - x) / width if centers[i] <= x < centers[i] + width
    #   0 if x >= centers[i] + width
    
    # This creates a set of overlapping triangles.
    # The sum of these hats is not constant 1, but it provides a flexible linear basis.
    # Since the model is linear in params, BFGS will solve the linear least squares problem efficiently.
    
    centers = np.linspace(-2.0, 2.0, 10)
    # Increase width to allow more overlap and smoother approximation
    # Original width was 1 interval. Let's try 2 intervals.
    width = 2.0 * (4.0 / 9.0)
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        c = centers[i]
        # Left part
        left = (x - (c - width)) / width
        left = np.clip(left, 0.0, 1.0)
        # Only active where x < c
        left_mask = x < c
        left = np.where(left_mask, left, 0.0)
        
        # Right part
        right = ((c + width) - x) / width
        right = np.clip(right, 0.0, 1.0)
        # Only active where x >= c
        right_mask = x >= c
        right = np.where(right_mask, right, 0.0)
        
        result += params[i] * (left + right)
        
    return result
# EVOLVE-BLOCK-END
