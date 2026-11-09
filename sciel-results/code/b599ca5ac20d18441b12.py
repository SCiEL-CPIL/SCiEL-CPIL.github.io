"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    High-capacity model combining:
    1. Chebyshev polynomial basis (degree 9) for smooth functions
    2. Gaussian RBF basis for localized features
    
    params[0]: log-scale width parameter for RBFs (sigma = exp(params[0]))
    params[1:10]: weights for the 9 Chebyshev terms
    (Note: params has length 10, so we use 9 Chebyshev + 1 RBF width)
    
    Wait, params has length 10. Let me restructure:
    params[0]: log-scale width for RBFs
    params[1:10]: 9 weights for Chebyshev polynomials T_0 to T_8
    
    Actually, let's use all 10 params more effectively:
    params[0]: log-scale width for RBFs (sigma = exp(params[0]))
    params[1:10]: 9 Chebyshev coefficients
    
    The RBF centers are fixed at 5 equally spaced points.
    The RBF contribution uses a fixed set of weights derived from the Chebyshev fit
    or we can make it simpler: just use Chebyshev + a small number of RBFs with
    shared width.
    
    Let me reconsider: with only 10 params, I should focus on a well-conditioned
    basis. Chebyshev polynomials of degree 9 give 10 coefficients (T_0 to T_9).
    That uses all 10 params. This is well-conditioned and can represent a wide
    class of smooth functions.
    
    Let's go with 10 Chebyshev coefficients (degree 0 to 9).
    """
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev
    # t = (2*x - (0.1 + 6)) / (6 - 0.1) = (2*x - 6.1) / 5.9
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)  # Safety clip
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    n_terms = 10
    T = np.zeros((n_terms, x.shape[0]))
    T[0, :] = 1.0
    if n_terms > 1:
        T[1, :] = t
    for k in range(2, n_terms):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # params[0:10] are the Chebyshev coefficients
    # prediction = sum_{k=0}^{9} params[k] * T_k(t)
    pred = T @ params[:10]
    
    return pred
# EVOLVE-BLOCK-END
