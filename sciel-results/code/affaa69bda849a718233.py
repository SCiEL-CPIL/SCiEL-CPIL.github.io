"""Symbolic regression with polynomial and sinusoidal basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using 10 parameters as coefficients of localized Gaussian basis functions.
    
    Basis functions: 10 Gaussian bumps centered at evenly spaced points in [-3, 3].
    Centers: -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 (only 7 fit in 10 params with fixed widths)
    
    Actually, let's use 10 centers evenly spaced: -3, -2.333, -1.667, -1, -0.333, 0.333, 1, 1.667, 2.333, 3
    Fixed width sigma = 0.8 for all Gaussians.
    
    p[i] is the amplitude of the Gaussian centered at c[i].
    """
    p = params
    # 10 centers evenly spaced from -3 to 3
    centers = np.linspace(-3.0, 3.0, 10)
    sigma = 0.8  # fixed width
    
    result = np.zeros_like(x)
    for i in range(10):
        result += p[i] * np.exp(-0.5 * ((x - centers[i]) / sigma) ** 2)
    
    return result
# EVOLVE-BLOCK-END
