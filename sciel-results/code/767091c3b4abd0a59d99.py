# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a physics-informed basis of bounded nonlinearities.
    Replaces high-degree polynomials with sin and tanh to capture saturation and periodicity
    while maintaining numerical stability for BFGS optimization.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute nonlinear functions
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Interaction terms using bounded nonlinearities
    xv = x * v
    x_sin_v = x * sin_v
    v_sin_x = v * sin_x
    
    # Construct the equation
    # Basis:
    # p0: bias
    # p1: sin(x) - primary restoring force (pendulum-like)
    # p2: x - linear restoring force
    # p3: tanh(x) - saturated restoring force
    # p4: v - linear damping
    # p5: sin(v) - nonlinear damping (small v approx linear)
    # p6: tanh(v) - saturated damping
    # p7: x*v - interaction
    # p8: x*sin(v) - interaction
    # p9: v*sin(x) - interaction
    
    return (p0 + 
            p1 * sin_x + 
            p2 * x + 
            p3 * tanh_x + 
            p4 * v + 
            p5 * sin_v + 
            p6 * tanh_v + 
            p7 * xv + 
            p8 * x_sin_v + 
            p9 * v_sin_x)
# EVOLVE-BLOCK-END
