"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# The evaluator calls equation(t, x, v, params).
# We define a robust nonlinear oscillator model.
# Structure: Fourier series in t + Polynomial in x and v with cross-terms.
# This covers Duffing (x^3), Van der Pol (x^2*v, v^3), and general periodic driving.
# Key insight: coupled terms (x*v, t*x, t*v) are critical for capturing nonlinearity.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.
    
    y = A(t, x, v) + B(x, v)
    
    A(t, x, v) = p0*sin(t) + p1*cos(t) + p2*sin(t)*x + p3*cos(t)*v
    B(x, v) = p4*x + p5*x^2 + p6*x^3 + p7*v + p8*x*v + p9*x^2*v
    
    Total 10 parameters.
    """
    # Pre-compute trigonometric functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    xv = x * v
    x2v = x2 * v
    
    # Coupled terms: time-dependent forcing modulated by state
    sin_t_x = sin_t * x
    cos_t_v = cos_t * v
    
    # Combine with parameters
    # p0, p1: Base driving force (Fourier series, 1st harmonics)
    # p2, p3: State-modulated driving (sin(t)*x, cos(t)*v)
    # p4, p5, p6: Spring force (Linear, Quadratic, Cubic)
    # p7: Linear damping
    # p8, p9: Nonlinear damping terms (x*v, x^2*v)
    
    y = (params[0] * sin_t + params[1] * cos_t +
         params[2] * sin_t_x + params[3] * cos_t_v +
         params[4] * x + params[5] * x2 + params[6] * x3 +
         params[7] * v + params[8] * xv + params[9] * x2v)
         
    return y
# EVOLVE-BLOCK-END
