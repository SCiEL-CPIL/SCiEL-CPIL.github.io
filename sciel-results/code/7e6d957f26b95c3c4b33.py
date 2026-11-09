"""Symbolic regression with learnable Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model using 10 learnable Gaussian RBF terms.
    
    Each RBF has a learnable center and width, allowing adaptation to
    local non-linear features at arbitrary positions.
    
    params (length 10):
    0: amplitude for RBF 1
    1: center for RBF 1
    2: width for RBF 1
    3: amplitude for RBF 2
    4: center for RBF 2
    5: width for RBF 2
    6: amplitude for RBF 3
    7: center for RBF 3
    8: width for RBF 3
    9: amplitude for RBF 4
    
    Wait, that's only 4 RBFs with 3 params each = 12 params, but we only have 10.
    Let me reconsider.
    
    Actually, let's use a different structure:
    - 5 RBFs, each with amplitude and center (width fixed)
    - Or: 2 RBFs with amplitude, center, width (6 params) + 4 polynomial terms
    
    Better approach: Use a hybrid basis:
    - 4 Chebyshev polynomials (T0 through T3) for global shape
    - 3 Gaussian RBFs with learnable centers and fixed widths
    - 3 additional RBFs with fixed centers and learnable amplitudes
    
    Total: 4 + 3*2 + 3 = 13... still too many.
    
    Let me use:
    - 3 Chebyshev polynomials (T0, T1, T2)
    - 2 Gaussian RBFs with learnable center and amplitude (width fixed at 1.0)
    - 2 Gaussian RBFs with learnable center and amplitude (width fixed at 0.5)
    - 1 Gaussian RBF with learnable center and amplitude (width fixed at 2.0)
    - 1 constant offset
    
    That's 3 + 2*2 + 2*2 + 1*2 + 1 = 3 + 4 + 4 + 2 + 1 = 14... still too many.
    
    Simplest approach: Use 10 RBFs with fixed centers spread across [-3,3], 
    and learnable amplitudes. But that's just linear regression on fixed basis.
    
    Better: Use a mixture of:
    - 5 RBFs with fixed centers at -3, -1.5, 0, 1.5, 3 and learnable amplitudes
    - 5 polynomial terms (1, x, x^2, x^3, x^4) with learnable coefficients
    
    Total: 10 params. This gives both local adaptation (RBFs) and global shape (polynomials).
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.ravel()
    
    # Polynomial basis: 1, x, x^2, x^3, x^4
    p0 = np.ones_like(x_flat)
    p1 = x_flat
    p2 = x_flat**2
    p3 = x_flat**3
    p4 = x_flat**4
    
    # RBF basis: Gaussian with fixed centers and width
    # Centers at -3, -1.5, 0, 1.5, 3 with width 1.5
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    width = 1.5
    
    rbf_terms = []
    for c in centers:
        rbf = np.exp(-((x_flat - c)**2) / (2.0 * width**2))
        rbf_terms.append(rbf)
    
    # Stack basis functions
    # Shape: (n_points, 10)
    basis = np.column_stack([p0, p1, p2, p3, p4] + rbf_terms)
    
    # params is length 10
    p = np.asarray(params, dtype=np.float64)
    
    # Prediction: basis @ params
    pred = basis @ p
    
    # Reshape to match input shape
    return pred.reshape(x.shape)
# EVOLVE-BLOCK-END
