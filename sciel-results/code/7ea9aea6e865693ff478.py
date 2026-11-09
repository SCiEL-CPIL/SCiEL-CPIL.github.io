"""Physics-informed parametric oscillator model.
Instead of a generic linear combination of basis functions, we use a structured
nonlinear form derived from Duffing oscillator dynamics. This reduces the
effective dimensionality of the optimization problem and provides BFGS with
a smoother, more well-conditioned loss landscape by aligning parameter roles
with physical quantities (stiffness, damping, nonlinearity)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction for a nonlinear oscillator.
    
    Structure:
    y = p0 + p1*x + p2*v + p3*x^2 + p4*x^3 + p5*x*v + p6*sin(x) + p7*t*x + p8*v^2 + p9*x^2*v
    
    This form captures:
    - Linear restoring force (p1*x)
    - Linear damping (p2*v)
    - Quadratic nonlinearity (p3*x^2)
    - Cubic Duffing nonlinearity (p4*x^3)
    - Velocity-position coupling (p5*x*v)
    - Periodic nonlinearity (p6*sin(x))
    - Time-dependent modulation (p7*t*x)
    - Velocity-squared damping (p8*v^2)
    - Higher-order coupling (p9*x^2*v)
    """
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute nonlinear terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    x2v = x2 * v
    
    # Combine terms
    pred = (p0 
            + p1 * x 
            + p2 * v 
            + p3 * x2 
            + p4 * x3 
            + p5 * xv 
            + p6 * np.sin(x) 
            + p7 * t * x 
            + p8 * v2 
            + p9 * x2v)
    
    return pred
# EVOLVE-BLOCK-END
