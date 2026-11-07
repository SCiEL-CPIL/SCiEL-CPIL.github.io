"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic Polynomial + 3 Gaussian RBFs.
    Parameters:
    p0, p1, p2, p3: Coefficients for 1, x, x^2, x^3
    p4, p5: Amplitude, WidthParam for RBF centered at -2
    p6, p7: Amplitude, WidthParam for RBF centered at 0
    p8, p9: Amplitude, WidthParam for RBF centered at 2
    """
    # Global polynomial trend
    poly = params[0] + params[1] * x + params[2] * x**2 + params[3] * x**3
    
    # RBF centers
    centers = np.array([-2.0, 0.0, 2.0])
    
    # Initialize output with polynomial part
    out = poly.copy()
    
    # RBF 1 (center -2)
    # Width parameterization: 0.1 + exp(p) to ensure positive width and avoid 0
    w1 = 0.1 + np.exp(params[5])
    a1 = params[4]
    rbf1 = a1 * np.exp(-0.5 * ((x - centers[0]) / w1) ** 2)
    out += rbf1
    
    # RBF 2 (center 0)
    w2 = 0.1 + np.exp(params[7])
    a2 = params[6]
    rbf2 = a2 * np.exp(-0.5 * ((x - centers[1]) / w2) ** 2)
    out += rbf2
    
    # RBF 3 (center 2)
    w3 = 0.1 + np.exp(params[9])
    a3 = params[8]
    rbf3 = a3 * np.exp(-0.5 * ((x - centers[2]) / w3) ** 2)
    out += rbf3
    
    return out
# EVOLVE-BLOCK-END
