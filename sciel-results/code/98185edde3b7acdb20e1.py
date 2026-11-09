"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Uses a sparse Gaussian RBF network with fixed centers and a quadratic polynomial trend.
    
    Parameters:
    p0: bias
    p1: coefficient for x
    p2: coefficient for v
    p3: coefficient for x^2
    p4: coefficient for v^2
    p5-p9: coefficients for 5 fixed RBFs
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial trend
    y_poly = p0 + p1 * x + p2 * v + p3 * (x ** 2) + p4 * (v ** 2)
    
    # RBF components
    # Centers chosen to cover the input domain:
    # x in [-0.6, 0.5], v in [-0.2, 0.3]
    # Centers:
    # 1: (-0.3, -0.1)
    # 2: (-0.3,  0.1)
    # 3: ( 0.2, -0.1)
    # 4: ( 0.2,  0.1)
    # 5: ( 0.0,  0.0)
    
    cx1, cv1 = -0.3, -0.1
    cx2, cv2 = -0.3,  0.1
    cx3, cv3 =  0.2, -0.1
    cx4, cv4 =  0.2,  0.1
    cx5, cv5 =  0.0,  0.0
    
    sigma = 0.25
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Calculate squared distances
    d1 = (x - cx1) ** 2 + (v - cv1) ** 2
    d2 = (x - cx2) ** 2 + (v - cv2) ** 2
    d3 = (x - cx3) ** 2 + (v - cv3) ** 2
    d4 = (x - cx4) ** 2 + (v - cv4) ** 2
    d5 = (x - cx5) ** 2 + (v - cv5) ** 2
    
    # Gaussian RBFs
    rbf1 = np.exp(-d1 * inv_2_sigma2)
    rbf2 = np.exp(-d2 * inv_2_sigma2)
    rbf3 = np.exp(-d3 * inv_2_sigma2)
    rbf4 = np.exp(-d4 * inv_2_sigma2)
    rbf5 = np.exp(-d5 * inv_2_sigma2)
    
    y_rbf = p5 * rbf1 + p6 * rbf2 + p7 * rbf3 + p8 * rbf4 + p9 * rbf5
    
    return y_poly + y_rbf
# EVOLVE-BLOCK-END
