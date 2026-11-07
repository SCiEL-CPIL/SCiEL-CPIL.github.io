"""Symbolic regression using Chebyshev polynomials + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid Chebyshev + RBF basis.
    
    Architecture:
    - 6 Chebyshev polynomial terms (T_0 to T_5) on scaled domain [-1, 1]
    - 4 Gaussian RBF terms centered at {-2.25, -0.75, 0.75, 2.25} with fixed width
    
    Parameters (length 10):
    - params[0:6]: Coefficients for the 6 Chebyshev terms
    - params[6:10]: Coefficients for the 4 RBF terms
    
    Chebyshev polynomials provide a well-conditioned global basis,
    while RBFs capture local non-smooth features. This combination
    is robust for smooth and non-smooth targets on [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev features: (N, 6)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    t2 = t**2
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    
    cheb = np.stack([
        np.ones_like(t),
        t,
        2 * t2 - 1,
        4 * t3 - 3 * t,
        8 * t4 - 8 * t2 + 1,
        16 * t5 - 20 * t3 + 5 * t
    ], axis=1)
    
    # Compute RBF features: (N, 4)
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    sigma = 0.5
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-diff**2 / (2 * sigma**2))
    
    # Unpack parameters
    cheb_coeffs = params[0:6]   # (6,)
    rbf_coeffs = params[6:10]   # (4,)
    
    # Compute output
    y = cheb @ cheb_coeffs + rbf @ rbf_coeffs
    
    return y
# EVOLVE-BLOCK-END
