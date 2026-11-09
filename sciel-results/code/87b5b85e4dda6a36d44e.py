"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _legendre_p0(t):
    return np.ones_like(t)

def _legendre_p1(t):
    return t

def _legendre_p2(t):
    return 0.5 * (3 * t**2 - 1)

def _legendre_p3(t):
    return 0.5 * (5 * t**3 - 3 * t)

def _legendre_p4(t):
    return 0.375 * (35 * t**4 - 30 * t**2 + 3)


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using shifted Legendre polynomials up to degree 3.
    
    Basis (10 terms):
    0: Constant (1)
    1: P1(x)
    2: P1(y)
    3: P1(z)
    4: P2(x)
    5: P2(y)
    6: P2(z)
    7: P3(x)
    8: P3(y)
    9: P3(z)
    
    This provides a well-conditioned cubic model in the Legendre basis,
    capturing higher-order curvature for each input variable.
    """
    # Normalize inputs to [-1, 1]
    # Input range [0.5, 2] -> center 1.25, half-range 0.75
    # t = (input - 1.25) / 0.75
    x_t = (x - 1.25) / 0.75
    y_t = (y - 1.25) / 0.75
    z_t = (z - 1.25) / 0.75
    
    # Compute Legendre polynomials up to degree 3
    x_p1 = _legendre_p1(x_t)
    x_p2 = _legendre_p2(x_t)
    x_p3 = _legendre_p3(x_t)
    
    y_p1 = _legendre_p1(y_t)
    y_p2 = _legendre_p2(y_t)
    y_p3 = _legendre_p3(y_t)
    
    z_p1 = _legendre_p1(z_t)
    z_p2 = _legendre_p2(z_t)
    z_p3 = _legendre_p3(z_t)
    
    # Combine with params (length 10)
    result = (params[0] * 1.0 +
              params[1] * x_p1 +
              params[2] * y_p1 +
              params[3] * z_p1 +
              params[4] * x_p2 +
              params[5] * y_p2 +
              params[6] * z_p2 +
              params[7] * x_p3 +
              params[8] * y_p3 +
              params[9] * z_p3)
    
    return result
# EVOLVE-BLOCK-END
