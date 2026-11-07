"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials + Trigonometric + Gaussian.
    
    Combines the stability and approximation power of Chebyshev polynomials
    (which can approximate any smooth function on [-1, 1] very efficiently)
    with trigonometric terms for periodic components and a Gaussian bump
    for localized features.
    
    The basis is:
    - Chebyshev T_0 to T_4 (5 terms): captures smooth polynomial-like behavior
    - sin(pi*t), cos(pi*t), sin(2*pi*t), cos(2*pi*t) (4 terms): captures periodic behavior
    - exp(-t^2) (1 term): captures a localized bump at t=0 (x=0)
    
    Total: 10 basis functions, linearly combined with 10 params.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Chebyshev polynomials T_0 ... T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Trigonometric terms scaled to the domain
    # Using pi*t so that t in [-1,1] maps to phase in [-pi, pi]
    # This gives one full period over the domain for the fundamental
    sin1 = np.sin(np.pi * t)
    cos1 = np.cos(np.pi * t)
    sin2 = np.sin(2.0 * np.pi * t)
    cos2 = np.cos(2.0 * np.pi * t)
    
    # Gaussian bump centered at t=0 (x=0)
    # Using t^2 for tighter localization
    b9 = np.exp(-t**2)
    
    # Linear combination
    # params is an array of length 10
    return (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            params[5] * sin1 +
            params[6] * cos1 +
            params[7] * sin2 +
            params[8] * cos2 +
            params[9] * b9)
# EVOLVE-BLOCK-END
