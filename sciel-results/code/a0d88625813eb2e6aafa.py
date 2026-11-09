"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rational function (Padé-like) basis.
    y = (p0 + p1*X + p2*V + p3*X^2 + p4*X*V) / (1 + p5*X^2 + p6*V^2 + p7*X^2*V^2 + p8*X^4)
    """
    # Scale inputs to [-1, 1] domain
    X = x / 0.6
    V = v / 0.3

    # Clip to [-1, 1] to avoid numerical issues
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Numerator terms
    num = (p0 + 
           p1 * X + 
           p2 * V + 
           p3 * X**2 + 
           p4 * X * V)

    # Denominator terms
    # We use 5 denominator parameters (p5-p9) but only 5 are needed for the 10 total params
    # Wait, I have 10 params. Let's use:
    # Numerator: p0, p1, p2, p3, p4 (5 terms)
    # Denominator: 1 + p5*X^2 + p6*V^2 + p7*X^2*V^2 + p8*X^4 + p9*V^4
    
    den = (1.0 + 
           p5 * X**2 + 
           p6 * V**2 + 
           p7 * X**2 * V**2 + 
           p8 * X**4 + 
           p9 * V**4)

    # Avoid division by zero
    den = np.where(np.abs(den) < 1e-8, np.sign(den) * 1e-8 + 1e-8, den)
    
    return num / den
# EVOLVE-BLOCK-END
