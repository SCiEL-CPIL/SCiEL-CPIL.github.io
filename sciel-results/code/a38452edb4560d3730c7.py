"""Rewritten program using Local Constant RBF + Global Quadratic model for nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Local Constant RBF model
    plus a Global Quadratic term.
    
    The model combines 4 Gaussian RBFs centered on a 2x2 grid in the (x, v) plane
    with a global quadratic polynomial. This allows capturing both smooth global
    trends and local nonlinear deviations.
    
    params[0:4]:  Weights for the 4 RBFs
    params[4]:    Global constant
    params[5]:    Global x coefficient
    params[6]:    Global v coefficient
    params[7]:    Global x^2 coefficient
    params[8]:    Global v^2 coefficient
    params[9]:    Global xv coefficient
    """
    # Define RBF centers (2x2 grid covering the training domain)
    # Training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    cx1, cv1 = -0.15, -0.20  # Bottom Left
    cx2, cv2 = 0.10, -0.20   # Bottom Right
    cx3, cv3 = -0.15, 0.25   # Top Left
    cx4, cv4 = 0.10, 0.25    # Top Right
    
    # RBF bandwidth
    sigma = 0.15
    
    # Compute RBF values
    rbf1 = np.exp(-((x - cx1)**2 + (v - cv1)**2) / (2 * sigma**2))
    rbf2 = np.exp(-((x - cx2)**2 + (v - cv2)**2) / (2 * sigma**2))
    rbf3 = np.exp(-((x - cx3)**2 + (v - cv3)**2) / (2 * sigma**2))
    rbf4 = np.exp(-((x - cx4)**2 + (v - cv4)**2) / (2 * sigma**2))
    
    # Local RBF contributions
    local_sum = params[0] * rbf1 + params[1] * rbf2 + params[2] * rbf3 + params[3] * rbf4
    
    # Global Quadratic contributions
    global_sum = (params[4] + 
                  params[5] * x + 
                  params[6] * v + 
                  params[7] * x**2 + 
                  params[8] * v**2 + 
                  params[9] * x * v)
    
    return local_sum + global_sum
# EVOLVE-BLOCK-END
