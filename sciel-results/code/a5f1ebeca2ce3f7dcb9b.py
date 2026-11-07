import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-center Gaussian RBF basis with two width scales.
    
    The 10 parameters are linear coefficients over a fixed basis of Gaussian
    Radial Basis Functions. 
    - 5 centers are placed uniformly across the domain [-2, 2].
    - Two width scales are used: a narrow scale (0.5) for local features and 
      a wide scale (1.5) for global trends.
    - This results in 5 * 2 = 10 basis functions.
    
    This structure allows the model to capture both localized variations and
    smooth global shapes, which is more robust than pure polynomials (which 
    can oscillate) or pure trigonometric series (which assume periodicity).
    BFGS fitting of linear parameters is convex and robust.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()

    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]

    # Define 5 centers uniformly distributed in [-2, 2]
    # -1.5, -0.75, 0.0, 0.75, 1.5
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    
    # Define 2 width scales
    # Narrow: 0.5 (sigma^2 = 0.25)
    # Wide: 1.5 (sigma^2 = 2.25)
    widths_sq_inv = np.array([1.0 / (2 * 0.5**2), 1.0 / (2 * 1.5**2)])
    # widths_sq_inv[0] = 1 / 0.5 = 2.0
    # widths_sq_inv[1] = 1 / 4.5 = 0.222...

    # Build basis matrix
    # We need 10 columns.
    # First 5: narrow RBFs
    # Last 5: wide RBFs
    
    basis_cols = []
    
    for c in centers:
        # Narrow RBF
        dist_sq = (x - c) ** 2
        rbf_narrow = np.exp(-widths_sq_inv[0] * dist_sq)
        basis_cols.append(rbf_narrow)
        
    for c in centers:
        # Wide RBF
        dist_sq = (x - c) ** 2
        rbf_wide = np.exp(-widths_sq_inv[1] * dist_sq)
        basis_cols.append(rbf_wide)
        
    basis = np.column_stack(basis_cols)
    
    result = basis @ p
    result = np.where(np.isfinite(result), result, 0.0)
    return result
