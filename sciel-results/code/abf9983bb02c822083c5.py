"""Symbolic regression with a richer, numerically stable basis and orthonormal polynomials."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 150s wall-clock
_DEADLINE = 145.0  # safety margin of 5s
_START_TIME = None


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    The evaluator fits `params` (length 10) via BFGS starting from all ones,
    minimizing training MSE. We design a flexible, smooth model with 10 parameters
    that can represent a wide variety of common function families.
    
    Parameter allocation (10 total):
      params[0..7]: Legendre polynomial coefficients (degree 0-7) on [-1,1]
      params[8]: amplitude of sin(x) (fixed frequency 1.0)
      params[9]: amplitude of cos(x) (fixed frequency 1.0)
    
    Using fixed-frequency trig terms and a higher-degree polynomial basis
    makes the model linear in parameters, yielding a well-conditioned
    quadratic optimization landscape that BFGS can navigate reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    elapsed = _get_elapsed()
    
    # Extract parameters
    c0, c1, c2, c3, c4, c5, c6, c7, amp_sin, amp_cos = params
    
    # Map x from [-3, 3] to [-1, 1] for standard Legendre
    t = x / 3.0
    
    # Legendre polynomials P0..P7 on [-1, 1]
    P0 = np.ones_like(t)
    P1 = t
    P2 = 0.5 * (3.0 * t**2 - 1.0)
    P3 = 0.5 * (5.0 * t**3 - 3.0 * t)
    P4 = (1.0/8.0) * (35.0 * t**4 - 30.0 * t**2 + 3.0)
    P5 = (1.0/16.0) * (63.0 * t**5 - 70.0 * t**3 + 15.0 * t)
    P6 = (1.0/32.0) * (231.0 * t**6 - 315.0 * t**4 + 105.0 * t**2 - 5.0)
    P7 = (1.0/64.0) * (429.0 * t**7 - 693.0 * t**5 + 315.0 * t**3 - 35.0 * t)
    
    # Polynomial part (linear in parameters)
    y_poly = (c0 * P0 + c1 * P1 + c2 * P2 + c3 * P3 +
              c4 * P4 + c5 * P5 + c6 * P6 + c7 * P7)
    
    # Fixed-frequency trigonometric terms (amplitudes are the parameters)
    y_sin = amp_sin * np.sin(x)
    y_cos = amp_cos * np.cos(x)
    
    # Combine all components
    result = y_poly + y_sin + y_cos
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
