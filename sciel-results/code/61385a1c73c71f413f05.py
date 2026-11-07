"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis using low-degree polynomials and Fourier terms.
    Basis: [1, t, t^2, t^3, t^4, sin(pi*t), cos(pi*t), sin(2*pi*t), cos(2*pi*t), sin(3*pi*t)]
    where t = x/2.
    This allows fitting both polynomial and periodic functions.
    """
    x = np.asarray(x, dtype=float)
    t = x / 2.0
    p = params
    
    # Polynomial terms
    poly_0 = np.ones_like(t)
    poly_1 = t
    poly_2 = t**2
    poly_3 = t**3
    poly_4 = t**4
    
    # Fourier terms (scaled to [-1, 1] domain)
    # Frequencies: 1, 1, 2, 2, 3 (cycles per 2 units of t, i.e., per 4 units of x)
    # sin(pi*t) has period 2 in t.
    # sin(2*pi*t) has period 1 in t.
    # sin(3*pi*t) has period 2/3 in t.
    
    four_0 = np.sin(np.pi * t)
    four_1 = np.cos(np.pi * t)
    four_2 = np.sin(2.0 * np.pi * t)
    four_3 = np.cos(2.0 * np.pi * t)
    four_4 = np.sin(3.0 * np.pi * t)
    
    result = (p[0] * poly_0 + 
              p[1] * poly_1 + 
              p[2] * poly_2 + 
              p[3] * poly_3 + 
              p[4] * poly_4 + 
              p[5] * four_0 + 
              p[6] * four_1 + 
              p[7] * four_2 + 
              p[8] * four_3 + 
              p[9] * four_4)
              
    return result
# EVOLVE-BLOCK-END
