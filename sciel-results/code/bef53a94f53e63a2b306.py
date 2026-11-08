# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with coupled time-state terms.
    
    Key structural change: includes coupled terms t*x and t*v which allow
    the model to capture parametric excitation and state-dependent damping
    that purely additive models cannot represent.
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Basis Functions:
    p0: Constant offset
    p1: Linear position (x)
    p2: Linear velocity (v)
    p3: Quadratic position (x^2)
    p4: Cubic position (x^3) - Duffing stiffness
    p5: Quadratic velocity (v^2)
    p6: Cross term x*v - state-dependent coupling
    p7: Coupled time-position term (t_c * x) - parametric excitation
    p8: Coupled time-velocity term (t_c * v) - time-varying damping
    p9: Sinusoidal forcing (sin(t_c))
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center time to improve numerical stability
    t_c = t - 40.0
    
    # Construct basis functions with coupled time-state terms
    f0 = np.ones_like(x)
    f1 = x
    f2 = v
    f3 = x ** 2
    f4 = x ** 3
    f5 = v ** 2
    f6 = x * v
    f7 = t_c * x
    f8 = t_c * v
    f9 = np.sin(t_c)
    
    # Linear combination of basis functions
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
