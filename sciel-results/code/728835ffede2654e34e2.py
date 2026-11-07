"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for acceleration of a nonlinear oscillator.
    Uses a 10-term linear basis with physically-motivated features:
    1, x, x^2, x^3, v, xv, x^2v, cos(t), sin(t), x*cos(t)
    
    Replaces v^2 (quadratic damping, often less important) with x*cos(t)
    to capture parametric resonance / drive-state interaction.
    """
    p = params
    
    # Precompute features
    x2 = x * x
    x3 = x2 * x
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    f0 = 1.0
    f1 = x
    f2 = x2
    f3 = x3
    f4 = v
    f5 = x * v
    f6 = x2 * v
    f7 = cos_t
    f8 = sin_t
    f9 = x * cos_t
    
    # Linear combination
    y = (p[0] * f0 +
         p[1] * f1 +
         p[2] * f2 +
         p[3] * f3 +
         p[4] * f4 +
         p[5] * f5 +
         p[6] * f6 +
         p[7] * f7 +
         p[8] * f8 +
         p[9] * f9)
         
    return y

# EVOLVE-BLOCK-END
