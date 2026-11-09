"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a Parametric Gaussian Mixture + Polynomial basis.
    
    This approach differs from fixed basis functions by allowing the centers and widths
    of the Gaussian components to be learned. This enables the model to adapt to local
    features at arbitrary locations within the input domain [-3, 3].
    
    Model:
    y = p0 + p1*x + p2*x^2 + p3*x^3 
        + p4 * exp(-((x - p5)^2) / (2 * max(p6, eps)^2))
        + p7 * exp(-((x - p8)^2) / (2 * max(p9, eps)^2))
        
    Parameters:
    p0-p3: Polynomial coefficients (offset, linear, quadratic, cubic)
    p4: Amplitude of Gaussian 1
    p5: Center of Gaussian 1
    p6: Width (sigma) of Gaussian 1
    p7: Amplitude of Gaussian 2
    p8: Center of Gaussian 2
    p9: Width (sigma) of Gaussian 2
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Gaussian parts
    # Use a small epsilon to avoid division by zero and ensure differentiability
    # np.where or np.clip is safer. Let's use np.maximum to keep sigma positive.
    # However, if sigma is negative, sigma^2 is same, but derivative wrt sigma changes sign?
    # sigma^2 in denominator. d/dsigma (1/sigma^2) = -2/sigma^3. 
    # If we use sigma^2 directly in the formula, the function is even in sigma.
    # BFGS might struggle if the landscape is symmetric and flat near 0.
    # Let's force sigma to be positive by squaring it in the denominator calculation?
    # exp(- (x-c)^2 / (2 * sigma^2))
    # If sigma is 0, this is problematic.
    
    eps = 1e-6
    sigma1_sq = np.maximum(p6 * p6, eps)
    sigma2_sq = np.maximum(p9 * p9, eps)
    
    gauss1 = p4 * np.exp(-((x - p5) ** 2) / (2.0 * sigma1_sq))
    gauss2 = p7 * np.exp(-((x - p8) ** 2) / (2.0 * sigma2_sq))
    
    return poly + gauss1 + gauss2
# EVOLVE-BLOCK-END
