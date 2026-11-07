"""Nonlinear oscillator acceleration via physics-informed polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a basis of common nonlinear terms.
    
    The 10 parameters are coefficients for:
    0: Constant (bias)
    1: Linear position (x)
    2: Linear velocity (v)
    3: Quadratic position (x^2)
    4: Cubic position (x^3) - Duffing nonlinearity
    5: State-dependent damping (x^2 * v) - Van der Pol nonlinearity
    6: Cubic velocity (v^3) - High-order damping/drag
    7: Cross-coupling (x * v)
    8: Sine driving (sin(t))
    9: Cosine driving (cos(t))
    
    This basis captures the essential dynamics of many nonlinear oscillators 
    (Duffing, Van der Pol, driven damped) and provides a smooth, 
    differentiable surface suitable for BFGS optimization.
    """
    # Compute nonlinear features
    x2 = x * x
    x3 = x2 * x
    v3 = v * v * v
    xv = x * v
    x2v = x2 * v
    
    # Periodic driving terms
    sint = np.sin(t)
    cost = np.cos(t)
    
    # Linear combination of basis functions
    return (
        params[0]
        + params[1] * x
        + params[2] * v
        + params[3] * x2
        + params[4] * x3
        + params[5] * x2v
        + params[6] * v3
        + params[7] * xv
        + params[8] * sint
        + params[9] * cost
    )
# EVOLVE-BLOCK-END
