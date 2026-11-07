"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a mixed polynomial and reciprocal basis.
    
    Rationale:
    - Previous attempts with pure polynomials (linear, quadratic) and pure exponentials 
      performed poorly (scores ~1.0 to ~7.4).
    - The score of 1.0 for linear suggests the model is just predicting the mean, implying 
      the function has significant non-linear structure that linear terms can't capture.
    - The jump to 7.4 for quadratic suggests either overfitting or that the true function 
      is not well-approximated by low-degree polynomials in the standard basis.
    - Reciprocal terms (1/x, 1/y, 1/z) are common in rational functions and can capture 
      hyperbolic-like behaviors or singularities that polynomials miss.
    - Combining linear terms with reciprocal terms provides a richer basis that can 
      approximate a wider class of functions, including rational functions of low order.
    - Adding ratio terms (x/y, y/z, z/x) captures multiplicative relationships and 
      anisotropic scaling that pure products or reciprocals might miss.
    
    Basis: 1, x, y, z, 1/x, 1/y, 1/z, x/y, y/z, z/x
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Prevent division by zero (though domain is [0.5, 2], this is a safety net)
    safe_x = np.where(np.abs(x) < 1e-12, 1e-12, x)
    safe_y = np.where(np.abs(y) < 1e-12, 1e-12, y)
    safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    
    # Compute basis elements
    inv_x = 1.0 / safe_x
    inv_y = 1.0 / safe_y
    inv_z = 1.0 / safe_z
    
    ratio_xy = x / safe_y
    ratio_yz = y / safe_z
    ratio_zx = z / safe_x
    
    # Linear combination
    pred = c0 + c1 * x + c2 * y + c3 * z + c4 * inv_x + c5 * inv_y + c6 * inv_z + c7 * ratio_xy + c8 * ratio_yz + c9 * ratio_zx
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.full_like(pred, 1e30)
        
    return pred
# EVOLVE-BLOCK-END
