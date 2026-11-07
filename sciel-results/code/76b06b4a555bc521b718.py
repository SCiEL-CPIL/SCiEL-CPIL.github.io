import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a flexible model with polynomial, sigmoid, and Gaussian components.
    
    The model combines:
    - A cubic polynomial for global trend
    - Two sigmoid (tanh) units for smooth step-like transitions
    - Two Gaussian bumps for localized peaks or wells
    
    This provides a rich function space that can approximate many common smooth functions.
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    a1 = params[4]
    s1 = params[5]
    c1 = params[6]
    
    a2 = params[7]
    s2 = params[8]
    c2 = params[9]
    
    # Polynomial part
    y = p0 + p1 * x + p2 * x * x + p3 * x * x * x
    
    # Sigmoid parts - use tanh for smooth saturation
    # Use softer clipping to maintain gradient flow during optimization
    s1_clip = np.tanh(s1 / 20.0) * 50.0
    s2_clip = np.tanh(s2 / 20.0) * 50.0
    
    # Shift centers slightly to help BFGS escape symmetric initialization
    c1_shifted = c1 + 0.1 * np.sin(p0)
    c2_shifted = c2 + 0.1 * np.cos(p1)
    
    y += a1 * np.tanh(s1_clip * (x - c1_shifted))
    y += a2 * np.tanh(s2_clip * (x - c2_shifted))
    
    return y
