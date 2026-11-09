# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses optimization to find positions that minimize the union area.
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    
    def compute_area(x):
        """Compute the area of the union of triangles given base positions x."""
        # Heights for quadrature
        m = np.arange(20000)
        y = (m + 0.5) / 20000.0
        w = (1.0 - y) / n  # width at each height
        
        # L_i at each height: shape (20000, 64)
        # L_i(y) = x_i + a_i * y
        L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
        
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        
        # Union length: w + sum of min(consecutive gaps, w)
        gaps = np.diff(L_sorted, axis=1)  # shape (20000, 63)
        union_lengths = w[:, np.newaxis] + np.sum(np.minimum(gaps, w[:, np.newaxis]), axis=1)
        
        # Area is the mean of union lengths
        return np.mean(union_lengths)
    
    # Initial guess: x_i = -a_i * 0.5 to center the spread
    x0 = -a * 0.5
    
    # Use L-BFGS-B for optimization
    result = minimize(
        compute_area,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    return result.x.astype(np.float64)


# EVOLVE-BLOCK-END
