import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using polynomial + Gaussian + tanh sigmoid components.
    
    Parameters (10 total):
    p0, p1, p2, p3: Polynomial coefficients (constant, linear, quadratic, cubic)
    p4, p5, p6: Gaussian 1 - amplitude, center, log-width
    p7, p8, p9: Tanh sigmoid - amplitude, center, log-width
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    # Gaussian bump: A * exp(-(x - mu)^2 / (2 * sigma^2))
    A1 = params[4]
    mu1 = params[5]
    sigma1 = np.exp(params[6])  # positive width via exp
    
    # Tanh sigmoid: A * tanh((x - c) / w)
    A2 = params[7]
    c2 = params[8]
    w2 = np.exp(params[9])  # positive width via exp
    
    # Polynomial part
    y = p0 + p1 * x + p2 * x * x + p3 * x * x * x
    
    # Gaussian part - clip exponent to avoid overflow
    diff1 = (x - mu1) / sigma1
    exponent = -0.5 * diff1 * diff1
    exponent = np.clip(exponent, -700, 0)  # avoid underflow issues
    y += A1 * np.exp(exponent)
    
    # Tanh sigmoid part
    y += A2 * np.tanh((x - c2) / w2)
    
    return y
