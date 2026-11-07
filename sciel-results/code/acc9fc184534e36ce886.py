"""Symbolic regression: recover hidden univariate f on [-3,3] with 10 free params.

Uses a 10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5} as a strictly
linear-in-params basis. The trigonometric functions are nearly orthogonal on [-3, 3],
creating a well-conditioned system that BFGS can solve reliably. This captures both
periodic and smooth non-periodic components of the target function.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term Fourier series:
    f(x) = p0 + p1*cos(x) + p2*sin(x) + p3*cos(2x) + p4*sin(2x)
           + p5*cos(3x) + p6*sin(3x) + p7*cos(4x) + p8*sin(4x) + p9*cos(5x)
    
    Linear in params, so BFGS finds the global least-squares minimum.
    The basis is well-conditioned due to near-orthogonality of trig functions.
    """
    p = params
    
    # Precompute trig values for frequencies 1-5
    cos1 = np.cos(x)
    sin1 = np.sin(x)
    cos2 = np.cos(2.0 * x)
    sin2 = np.sin(2.0 * x)
    cos3 = np.cos(3.0 * x)
    sin3 = np.sin(3.0 * x)
    cos4 = np.cos(4.0 * x)
    sin4 = np.sin(4.0 * x)
    cos5 = np.cos(5.0 * x)
    
    # Linear combination with constant term first
    y = (p[0] + 
         p[1] * cos1 + p[2] * sin1 +
         p[3] * cos2 + p[4] * sin2 +
         p[5] * cos3 + p[6] * sin3 +
         p[7] * cos4 + p[8] * sin4 +
         p[9] * cos5)
    
    return y
# EVOLVE-BLOCK-END
