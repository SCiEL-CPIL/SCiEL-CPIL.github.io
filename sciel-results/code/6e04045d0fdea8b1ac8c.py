# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses a structured model with physically-motivated terms:
    - Linear restoring: p0 * x
    - Linear damping: p1 * v
    - Quadratic position: p2 * x^2
    - Quadratic velocity: p3 * v^2
    - Cross quadratic: p4 * x * v
    - Cubic position (Duffing): p5 * x^3
    - Cubic velocity: p6 * v^3
    - Cubic cross x^2*v: p7 * x^2 * v
    - Cubic cross x*v^2: p8 * x * v^2
    - Constant offset: p9
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Structured polynomial with physical interpretation
    a = (p[0] * x +
         p[1] * v +
         p[2] * x**2 +
         p[3] * v**2 +
         p[4] * x * v +
         p[5] * x**3 +
         p[6] * v**3 +
         p[7] * x**2 * v +
         p[8] * x * v**2 +
         p[9])
    
    return a
# EVOLVE-BLOCK-END
