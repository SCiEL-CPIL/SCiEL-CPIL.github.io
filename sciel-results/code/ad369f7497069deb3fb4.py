"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    Uses a smooth, physically-motivated basis including odd powers, 
    hyperbolic tangents for saturation effects, and sinusoidal terms.
    """
    p = params
    
    # Pre-compute smooth basis functions
    # Odd powers for restoring force and damping
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v3 = v2 * v
    
    # Interaction terms
    xv = x * v
    
    # Smooth non-linear saturating functions
    # tanh approximates softening for large inputs, linear for small
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # Sinusoidal term for periodic non-linearity
    sin_x = np.sin(x)
    
    # Linear combination of features
    # 0: x (linear restoring)
    # 1: x^3 (cubic restoring)
    # 2: x^5 (quintic restoring)
    # 3: v (linear damping)
    # 4: v^3 (cubic damping)
    # 5: x*v (parametric coupling)
    # 6: tanh(x) (softening stiffness)
    # 7: tanh(v) (softening damping)
    # 8: sin(x) (smooth periodic stiffness)
    # 9: 1 (bias)
    
    y = (p[0] * x +
         p[1] * x3 +
         p[2] * x5 +
         p[3] * v +
         p[4] * v3 +
         p[5] * xv +
         p[6] * tanh_x +
         p[7] * tanh_v +
         p[8] * sin_x +
         p[9])
         
    return y
# EVOLVE-BLOCK-END
