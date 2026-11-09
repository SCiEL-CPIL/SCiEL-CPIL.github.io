"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a hybrid polynomial + trigonometric basis.
    
    Physics motivation:
    - Polynomial terms capture local Taylor expansion of nonlinear restoring/damping
    - Trigonometric terms capture global oscillatory behavior (sin for position, 
      phase-like for velocity interactions)
    - Interaction terms couple position and velocity nonlinearly
    
    params layout (10 parameters):
    0: bias (constant force offset)
    1: linear restoring force coefficient (x)
    2: linear damping coefficient (v)
    3: cubic restoring force coefficient (x^3)
    4: cubic damping coefficient (v^3)
    5: position-velocity cross term (x*v)
    6: quadratic restoring (x^2)
    7: sin(x) - periodic restoring component
    8: sin(v) - periodic velocity modulation
    9: x^2 * v - nonlinear drag/interaction
    """
    p = params
    
    # Core polynomial terms
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    # Trigonometric terms for global oscillatory behavior
    sin_x = np.sin(x)
    sin_v = np.sin(v)
    
    return (p[0] +
            p[1] * x +
            p[2] * v +
            p[3] * x3 +
            p[4] * v3 +
            p[5] * x * v +
            p[6] * x2 +
            p[7] * sin_x +
            p[8] * sin_v +
            p[9] * x2 * v)
# EVOLVE-BLOCK-END
