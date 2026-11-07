"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a hybrid basis:
    - Polynomial terms for local dynamics (Duffing, Van der Pol)
    - Gaussian RBFs in (x, v) space for localized nonlinear response
    - Sinusoidal driving with fitted frequency
    
    params layout (10 constants):
    p0: constant offset
    p1: linear x coefficient
    p2: cubic x coefficient
    p3: linear v coefficient
    p4: x^2*v coefficient
    p5: driving frequency omega
    p6: sin(omega*t) amplitude
    p7: cos(omega*t) amplitude
    p8: x*v coefficient
    p9: x^2 coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Polynomial basis for local oscillator dynamics
    x2 = x * x
    x3 = x2 * x
    x2v = x2 * v
    xv = x * v

    # Sinusoidal driving terms
    omega = p5
    st = np.sin(omega * t)
    ct = np.cos(omega * t)

    # Gaussian RBF terms in (x, v) space
    # Centers and widths chosen to cover the training ranges
    # x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    # We'll use 2 RBFs with different centers to capture localized behavior
    
    # RBF 1: centered near (0, 0) - captures behavior near equilibrium
    c1x, c1v = 0.0, 0.0
    w1x, w1v = 0.15, 0.25
    rbf1 = np.exp(-((x - c1x) / w1x) ** 2 - ((v - c1v) / w1v) ** 2)
    
    # RBF 2: centered near positive x, negative v
    c2x, c2v = 0.1, -0.2
    w2x, w2v = 0.12, 0.20
    rbf2 = np.exp(-((x - c2x) / w2x) ** 2 - ((v - c2v) / w2v) ** 2)
    
    # RBF 3: centered near negative x, positive v
    c3x, c3v = -0.1, 0.2
    w3x, w3v = 0.12, 0.20
    rbf3 = np.exp(-((x - c3x) / w3x) ** 2 - ((v - c3v) / w3v) ** 2)

    # Combine all terms
    # Use p8 and p9 for RBF coefficients, and add a small weight to existing polynomial terms
    # This keeps the total number of parameters at 10
    
    return (
        p0
        + p1 * x
        + p2 * x3
        + p3 * v
        + p4 * x2v
        + p6 * st
        + p7 * ct
        + p8 * (xv + 0.5 * rbf1 + 0.3 * rbf2 + 0.3 * rbf3)
        + p9 * (x2 + 0.5 * rbf1 + 0.3 * rbf2 + 0.3 * rbf3)
    )
# EVOLVE-BLOCK-END
