"""
Physically-motivated nonlinear oscillator model.
Uses a Duffing-van der Pol style structure with sinusoidal driving.
This constrains the 10 parameters to specific physical roles,
reducing overfitting compared to a generic polynomial.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts acceleration for a nonlinear oscillator.
    
    Model: a = -k*x - c*v + alpha*x^3 + beta*v^2 + gamma*x*v + F*sin(w*t)
    Since we don't have time 't' directly, we approximate the driving force
    using a phase-like term derived from x and v, or simply a constant drive
    if the system is autonomous. However, standard symbolic regression for
    oscillators often assumes autonomous systems or uses x,v as state.
    
    Given the inputs are x and v, and no time t, we assume an autonomous
    system or that the 'driving' is implicit. A common autonomous nonlinear
    oscillator form is the Duffing-van der Pol equation:
    x'' = mu*(1 - x^2)*x' - x - x^3
    
    We extend this with additional cross-terms to utilize all 10 parameters
    while maintaining physical plausibility:
    a = p0*x + p1*v + p2*x^3 + p3*v^2 + p4*x*v + p5*x^2*v + p6*v*x^2 
      + p7*sin(x) + p8*cos(v) + p9
    
    To ensure the optimizer doesn't get stuck in bad local minima due to
    symmetry or scale issues, we normalize the inputs slightly within the
    function if necessary, but the task says inputs are in range.
    
    Let's use a robust mix of polynomial and trigonometric terms to capture
    nonlinearity without the high variance of high-degree polynomials.
    """
    p = params
    
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    c4 = p[4]
    c5 = p[5]
    c6 = p[6]
    c7 = p[7]
    c8 = p[8]
    c9 = p[9]
    
    # Precompute terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    
    # Physical-inspired terms:
    # Linear restoring force (Spring)
    term_linear_x = c0 * x
    
    # Linear damping
    term_linear_v = c1 * v
    
    # Nonlinear stiffness (Duffing)
    term_cubic_x = c2 * x3
    
    # Nonlinear damping (Van der Pol like, but we don't have x^2*v directly 
    # as a single simple term without mixing, so let's use v^2 and x*v)
    term_quad_v = c3 * v2
    
    # Cross coupling
    term_xv = c4 * xv
    
    # Higher order cross terms to add flexibility
    term_x2v = c5 * x2 * v
    term_xv2 = c6 * x * v2
    
    # Saturation/Softening effects via trigonometric functions
    # These are bounded and smooth, helping generalization
    term_sin_x = c7 * np.sin(x)
    term_cos_v = c8 * np.cos(v)
    
    # Constant offset (DC bias or mean acceleration)
    term_const = c9
    
    return term_linear_x + term_linear_v + term_cubic_x + term_quad_v + term_xv + term_x2v + term_xv2 + term_sin_x + term_cos_v + term_const
# EVOLVE-BLOCK-END
