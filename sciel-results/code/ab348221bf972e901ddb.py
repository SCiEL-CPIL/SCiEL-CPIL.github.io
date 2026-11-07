"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single Neuron Tanh Model for nonlinear oscillator acceleration.
    
    Form:
    a = p[9] * tanh(p[0] + p[1]*sin(t) + p[2]*cos(t) + p[3]*x + p[4]*v + 
                    p[5]*x^2 + p[6]*v^2 + p[7]*x*v + p[8]*x^3)
    
    This structure uses a hyperbolic tangent activation to model saturation 
    effects inherent in nonlinear oscillators, bounded by [-p[9], p[9]].
    The arguments include linear, quadratic, and cubic terms in position x,
    linear terms in velocity v, and periodic drive terms in t.
    """
    p = params
    
    # Compute the argument for the tanh function
    # Using sin(t) and cos(t) to handle the periodic time dependence
    # x and v are small, so higher order terms are small but capture nonlinearity
    arg = (p[0] 
           + p[1] * np.sin(t) 
           + p[2] * np.cos(t) 
           + p[3] * x 
           + p[4] * v 
           + p[5] * x**2 
           + p[6] * v**2 
           + p[7] * x * v 
           + p[8] * x**3)
           
    # Apply tanh and scale
    # p[9] acts as the maximum amplitude / scale
    return p[9] * np.tanh(arg)
# EVOLVE-BLOCK-END
