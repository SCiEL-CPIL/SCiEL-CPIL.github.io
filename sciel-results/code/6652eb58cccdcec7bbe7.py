"""Symbolic regression with a richer, numerically stable basis and orthonormal polynomials."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    The evaluator fits `params` (length 10) via BFGS starting from all ones,
    minimizing training MSE. We design a flexible, smooth model with 10 parameters
    that can represent a wide variety of common function families.
    
    Parameter allocation (10 total):
      params[0]: constant offset
      params[1]: linear coefficient
      params[2]: quadratic coefficient (scaled)
      params[3]: cubic coefficient (scaled)
      params[4]: amplitude of sin(freq1 * x)
      params[5]: frequency of sin(freq1 * x)
      params[6]: amplitude of cos(freq2 * x)
      params[7]: frequency of cos(freq2 * x)
      params[8]: rate of exp(rate * x)
      params[9]: amplitude of exp(rate * x)
    
    All parameters are used in a smooth, differentiable expression suitable for BFGS.
    Frequencies and rates are clipped to avoid numerical overflow/underflow.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    offset = params[0]
    a_lin = params[1]
    a_quad = params[2]
    a_cub = params[3]
    amp_sin = params[4]
    freq_sin = params[5]
    amp_cos = params[6]
    freq_cos = params[7]
    rate_exp = params[8]
    amp_exp = params[9]
    
    # Clip frequencies and rates for numerical stability
    freq_sin = np.clip(freq_sin, -15, 15)
    freq_cos = np.clip(freq_cos, -15, 15)
    rate_exp = np.clip(rate_exp, -5, 5)
    
    # Use orthonormal Legendre polynomials scaled to [-3, 3]
    # Map x from [-3, 3] to [-1, 1] for standard Legendre
    t = x / 3.0  # now in [-1, 1]
    
    # Legendre polynomials (up to degree 3)
    P0 = np.ones_like(t)
    P1 = t
    P2 = 0.5 * (3 * t**2 - 1)
    P3 = 0.5 * (5 * t**3 - 3 * t)
    
    # Scale coefficients to account for orthonormality
    # The norm of Pn on [-1,1] is sqrt(2/(2n+1))
    # So Pn / sqrt(2/(2n+1)) is orthonormal
    # But we just use the standard polynomials with learnable coefficients
    
    # Polynomial terms using Legendre basis
    y_polynomial = offset + a_lin * P1 + a_quad * P2 + a_cub * P3
    
    # Trigonometric terms
    y_sin = amp_sin * np.sin(freq_sin * x)
    y_cos = amp_cos * np.cos(freq_cos * x)
    
    # Exponential term with clipping for stability
    exp_arg = rate_exp * x
    exp_arg = np.clip(exp_arg, -20, 20)
    y_exp = amp_exp * np.exp(exp_arg)
    
    # Combine all components
    result = y_polynomial + y_sin + y_cos + y_exp
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
