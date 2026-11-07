"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear oscillator acceleration.
    
    This model combines the best-performing structural features from previous iterations:
    - Raw input scales (avoiding normalization which degraded performance).
    - Duffing-type nonlinear restoring (x, x^3).
    - Van der Pol style nonlinear damping (v, x^2*v).
    - Quadratic drag (v^2) and position correction (x^2).
    - Parametric coupling (x*v).
    - Full sinusoidal drive (sin(t), cos(t)) to account for unknown phase.
    - Constant offset.
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants [p0..p9]
    """
    p = params
    
    # Precompute powers and products for efficiency
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    # Core oscillator terms mapped to params
    # p0: Linear stiffness coefficient (x)
    # p1: Cubic (Duffing) stiffness coefficient (x^3)
    # p2: Linear damping coefficient (v)
    # p3: Nonlinear (Van der Pol) damping coefficient (x^2 * v)
    # p4: Quadratic position correction (x^2)
    # p5: Velocity squared (quadratic drag) coefficient (v^2)
    # p6: Cross-coupling term coefficient (x * v)
    # p7: Sinusoidal drive amplitude (sin(t))
    # p8: Cosinusoidal drive amplitude (cos(t))
    # p9: Constant offset
    
    # Calculate terms
    term_0 = p[0] * x
    term_1 = p[1] * x3
    term_2 = p[2] * v
    term_3 = p[3] * x2 * v
    term_4 = p[4] * x2
    term_5 = p[5] * v2
    term_6 = p[6] * x * v
    term_7 = p[7] * np.sin(t)
    term_8 = p[8] * np.cos(t)
    term_9 = p[9]
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
# EVOLVE-BLOCK-END
