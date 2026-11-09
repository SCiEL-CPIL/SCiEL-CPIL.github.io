"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Padé-like rational approximation.
    
    Form: y = (p0 + p1*v + p2*v^2 + p3*v^3 + p4*x) / (1 + p5*v^2 + p6*x^2 + p7*x*v + p8*v^4 + p9*x^2*v^2)
    
    The denominator provides inherent saturation and better conditioning for BFGS
    by preventing unbounded growth of higher-order terms, which is beneficial for
    capturing smooth, bounded nonlinear dynamics.
    
    params[0..4]: numerator coefficients
    params[5..9]: denominator coefficients (added to 1)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator: linear in x, cubic in v
    numerator = p0 + p1 * v + p2 * v**2 + p3 * v**3 + p4 * x
    
    # Denominator: 1 + quadratic/cubic terms for saturation
    # Keep terms moderate to avoid numerical instability
    v2 = v * v
    x2 = x * v * 0 + x * x  # x^2
    denom = 1.0 + p5 * v2 + p6 * x2 + p7 * x * v + p8 * v2 * v2 + p9 * x2 * v2
    
    # Avoid division by zero or near-zero denominators
    # Clip denominator to ensure it stays away from zero
    denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6 * np.sign(denom + 1e-15), denom)
    
    return numerator / denom_safe
# EVOLVE-BLOCK-END
