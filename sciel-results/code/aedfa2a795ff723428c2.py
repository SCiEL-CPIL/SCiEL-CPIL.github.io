"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a Rational Function approach.
    
    Model:
    a = p0 + (p1*x + p2*v + p3*x*v) / (1 + p4*x^2 + p5*v^2 + p6*x*v) + p7*sin(p8*t + p9)
    
    Parameters:
    p0: Constant bias
    p1: Linear position coefficient (numerator)
    p2: Linear velocity coefficient (numerator)
    p3: Position-velocity coupling (numerator)
    p4: Quadratic position coefficient (denominator)
    p5: Quadratic velocity coefficient (denominator)
    p6: Position-velocity coupling (denominator)
    p7: Drive amplitude
    p8: Drive frequency
    p9: Drive phase
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator of the rational term
    num = p1 * x + p2 * v + p3 * x * v
    
    # Denominator of the rational term
    # Adding 1 ensures the denominator is non-zero near the origin and stable for small p4, p5, p6
    den = 1.0 + p4 * x**2 + p5 * v**2 + p6 * x * v
    
    # Rational term representing nonlinear restoring/damping
    rational_term = num / den
    
    # Driving force
    drive = p7 * np.sin(p8 * t + p9)
    
    # Total acceleration
    return p0 + rational_term + drive
# EVOLVE-BLOCK-END
