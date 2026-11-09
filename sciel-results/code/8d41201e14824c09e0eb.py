import numpy as np
from numpy.polynomial.chebyshev import chebvander
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (degrees 0-6) + learnable sigmoid + two Gaussian RBFs.
    
    Parameters (10 total):
    0-6: Chebyshev coefficients for T_0 through T_6
    7: Sigmoid amplitude
    8: Sigmoid center
    9: Sigmoid steepness
    
    The Chebyshev basis provides a stable, well-conditioned polynomial approximation
    for smooth global structure. The sigmoid captures step-like or saturating behaviors.
    The Gaussians (with fixed width) capture local bumps.
    
    Wait, I only have 10 params. Let me restructure:
    0-5: Chebyshev coefficients for T_0 through T_5 (6 params)
    6: Sigmoid amplitude
    7: Sigmoid center
    8: Sigmoid steepness
    9: Gaussian amplitude (with fixed center at 0, fixed width)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    t0 = time.time()
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev basis values of degree 0 to 5
    # chebvander returns a matrix of shape (len(x), 6)
    # Columns correspond to T_0, T_1, ..., T_5
    basis = chebvander(t, 5)
    
    # Linear combination of Chebyshev terms
    cheb_part = basis @ p[:6]
    
    # Sigmoid part: A * sigmoid(k * (x - c))
    A_sig = p[6]
    c_sig = p[7]
    k_sig = p[8]
    
    z = k_sig * (x - c_sig)
    z_clipped = np.clip(z, -50, 50)
    sigmoid = A_sig * (1.0 / (1.0 + np.exp(-z_clipped)))
    
    # Gaussian RBF with fixed center at 0 and fixed width 0.5
    A_g = p[9]
    width = 0.5
    gaussian = A_g * np.exp(-(x ** 2) / (2.0 * width ** 2))
    
    result = cheb_part + sigmoid + gaussian
    
    return result
