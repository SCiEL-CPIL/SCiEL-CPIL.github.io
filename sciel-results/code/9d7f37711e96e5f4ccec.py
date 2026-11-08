"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# The evaluator calls equation(t, x, v, params).
# We define a robust nonlinear oscillator model with coupled time-state terms.
# Structure: Forcing + Spring + Damping + Parametric coupling.
# This covers Duffing (x^3), Van der Pol (x^2*v), and parametric excitation.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.
    
    y = Forcing(t) + Spring(x) + Damping(v, x) + Parametric(t, x, v)
    
    p0*sin(t) + p1*cos(t) + p2*x + p3*x^3 + p4*v + p5*x*v + p6*sin(t)*x + p7*cos(t)*x + p8*sin(t)*v + p9*x^2*v
    
    Total 10 parameters.
    """
    # Pre-compute trigonometric functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # State terms
    x2 = x * x
    x3 = x2 * x
    xv = x * v
    x2v = x2 * v
    
    # Coupled time-state terms (parametric excitation)
    st_x = sin_t * x
    ct_x = cos_t * x
    st_v = sin_t * v
    
    # Combine with parameters
    # p0, p1: External forcing (1st harmonics)
    # p2: Linear spring
    # p3: Nonlinear spring (Duffing)
    # p4: Linear damping
    # p5: State-dependent damping (Van der Pol-like)
    # p6, p7: Parametric forcing on position
    # p8: Parametric forcing on velocity
    # p9: Higher-order damping (Van der Pol)
    
    y = (params[0] * sin_t + params[1] * cos_t +
         params[2] * x + params[3] * x3 +
         params[4] * v + params[5] * xv +
         params[6] * st_x + params[7] * ct_x +
         params[8] * st_v + params[9] * x2v)
         
    return y
# EVOLVE-BLOCK-END
