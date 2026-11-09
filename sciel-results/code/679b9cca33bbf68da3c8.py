"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a Gaussian Radial Basis Function (RBF) network.
    
    Structural change from the current polynomial/trig/exponential basis:
    Uses 9 fixed Gaussian kernels centered evenly across the input domain [0.1, 6],
    plus one global constant term. The 9 kernel amplitudes are learned by BFGS.
    
    This provides a localized, non-polynomial basis that can sharply approximate
    non-smooth or complex hidden functions better than global Chebyshev expansions.
    """
    x = np.asarray(x, dtype=float)
    
    # 9 fixed RBF centers spanning the domain [0.1, 6]
    centers = np.linspace(0.1, 6.0, 9)
    
    # Fixed Gaussian width (sigma) - chosen to balance locality and coverage
    # Domain width is ~5.9, 9 centers give spacing ~0.7375
    # sigma ~ 0.8 gives good overlap between adjacent kernels
    sigma = 0.8
    
    # Compute Gaussian kernels: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # x has shape (n,), centers has shape (9,)
    # We want output shape (n, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponents = -0.5 * (diff / sigma) ** 2
    kernels = np.exp(exponents)
    
    # params has length 10: 9 amplitudes for RBFs + 1 constant
    # p0..p8 are RBF amplitudes, p9 is the constant term
    y = kernels @ params[:9] + params[9]
    
    return y
# EVOLVE-BLOCK-END
