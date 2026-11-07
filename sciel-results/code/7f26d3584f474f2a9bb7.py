"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a combination of polynomial, trigonometric,
    and hyperbolic basis functions to recover an unknown function.
    
    The expression is designed to be flexible enough to fit a wide variety
    of functions while remaining differentiable for BFGS optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis up to degree 3
    # This captures general trends and odd/even symmetries
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Trigonometric basis with parametric frequencies
    # Clip frequencies to avoid extreme high-frequency oscillations that might
    # hinder BFGS convergence due to large gradients, while allowing enough
    # flexibility for periodic functions in [-2, 2].
    freq_sin = np.clip(p5, -10, 10)
    freq_cos = np.clip(p7, -10, 10)
    
    trig_sin = p4 * np.sin(freq_sin * x)
    trig_cos = p6 * np.cos(freq_cos * x)
    
    # Hyperbolic tangent basis with parametric scale
    # Captures sigmoidal or saturation behaviors
    scale_tanh = np.clip(p9, -10, 10)
    tanh_term = p8 * np.tanh(scale_tanh * x)
    
    return poly + trig_sin + trig_cos + tanh_term
# EVOLVE-BLOCK-END
