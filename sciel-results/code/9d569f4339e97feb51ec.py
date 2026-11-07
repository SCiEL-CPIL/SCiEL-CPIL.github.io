"""
Strategy: Learnable Gaussian Mixture (2 components) + Cubic Polynomial.
This approach uses 6 parameters to define two adaptive Gaussian bumps 
(centers, widths, amplitudes) and 4 parameters for a cubic polynomial baseline.
This allows the model to capture both local non-linear features and global trends,
which is more flexible than a fixed basis of polynomials and trig functions.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    
    # Unpack parameters
    # 2 Gaussians: center, width, amplitude for each
    c1 = params[0]
    w1 = params[1]**2 + 0.1  # Ensure positive width, smooth
    a1 = params[2]
    
    c2 = params[3]
    w2 = params[4]**2 + 0.1  # Ensure positive width, smooth
    a2 = params[5]
    
    # 4 Coefficients for cubic polynomial: p0 + p1*x + p2*x^2 + p3*x^3
    p0 = params[6]
    p1 = params[7]
    p2 = params[8]
    p3 = params[9]
    
    # Compute Gaussians
    # Using standard Gaussian form: exp(-(x-c)^2 / (2*w^2))
    g1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    g2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    
    # Compute Polynomial
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Combine
    return g1 + g2 + poly
# EVOLVE-BLOCK-END
