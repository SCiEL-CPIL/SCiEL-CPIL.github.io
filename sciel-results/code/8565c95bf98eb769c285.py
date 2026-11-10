import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis expansion: polynomial, trigonometric, and exponential basis functions.
    
    Parameters (10 total):
      - params[0]: weight for constant basis (1)
      - params[1]: weight for linear basis (x)
      - params[2]: weight for quadratic basis (x^2)
      - params[3]: weight for cubic basis (x^3)
      - params[4]: weight for sin(x)
      - params[5]: weight for cos(x)
      - params[6]: weight for exp(-x/3) (scaled to be O(1) on [-3,3])
      - params[7]: weight for exp(x/3)  (scaled to be O(1) on [-3,3])
      - params[8]: weight for tanh(x)
      - params[9]: weight for 1/(1+x^2) (Lorentzian)
    
    This is a linear-in-params model, so BFGS will converge to the global optimum.
    The diverse basis functions span a wide variety of common functional forms.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Polynomial basis
    x2 = x * x
    x3 = x2 * x
    
    # Trigonometric basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Exponential basis - scaled to keep values O(1) on [-3, 3]
    # exp(-x/3) ranges from exp(-1)≈0.37 to exp(1)≈2.72
    # exp(x/3) ranges from exp(-1)≈0.37 to exp(1)≈2.72
    exp_neg = np.exp(-x / 3.0)
    exp_pos = np.exp(x / 3.0)
    
    # Hyperbolic tangent basis
    tanh_x = np.tanh(x)
    
    # Lorentzian basis
    lorentz = 1.0 / (1.0 + x2)
    
    # Combine all basis functions
    result = (
        c0 * np.ones_like(x) +
        c1 * x +
        c2 * x2 +
        c3 * x3 +
        c4 * sin_x +
        c5 * cos_x +
        c6 * exp_neg +
        c7 * exp_pos +
        c8 * tanh_x +
        c9 * lorentz
    )
    
    return result
