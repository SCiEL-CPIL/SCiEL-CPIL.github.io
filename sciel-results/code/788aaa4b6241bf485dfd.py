# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a physically-motivated basis.
    
    The model combines:
    1. Polynomial terms in x and v to capture nonlinear stiffness (Duffing) 
       and nonlinear damping (Van der Pol).
    2. Trigonometric terms in centered time t to capture periodic forcing.
    
    Inputs:
    t: Time (centered internally for numerical stability)
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Basis Functions:
    p0: Constant offset
    p1: Linear position (x)
    p2: Linear velocity (v)
    p3: Quadratic position (x^2)
    p4: Cubic position (x^3) - Duffing term
    p5: Quadratic velocity (v^2)
    p6: Cross term x^2*v - Van der Pol damping term
    p7: Cross term x*v
    p8: Sinusoidal forcing (sin(t_c))
    p9: Cosinusoidal forcing (cos(t_c))
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center time to improve numerical stability and phase representation
    # Midpoint of [30, 50] is 40
    t_c = t - 40.0
    
    # Construct basis functions
    f0 = np.ones_like(x)
    f1 = x
    f2 = v
    f3 = x ** 2
    f4 = x ** 3
    f5 = v ** 2
    f6 = x ** 2 * v
    f7 = x * v
    f8 = np.sin(t_c)
    f9 = np.cos(t_c)
    
    # Linear combination of basis functions
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
