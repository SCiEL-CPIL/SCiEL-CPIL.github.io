# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-motivated model
    with tanh saturation for nonlinear stiffness and damping.
    
    Model: a = p0*tanh(x/s1) + p1*x^3 + p2*tanh(v/t1) + p3*v^3
             + p4*x*v + p5*x^2*v + p6*v^2*x + p7*sin(w1*x) + p8*cos(w2*v) + p9
    
    where s1, t1, w1, w2 are fixed scaling constants chosen for conditioning.
    The tanh terms provide bounded nonlinearities that capture saturation
    dynamics typical in oscillators, while polynomial terms capture local
    behavior.
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Fixed scaling constants for conditioning
    s1 = 0.5   # saturation scale for position
    t1 = 0.2   # saturation scale for velocity
    w1 = 2.0   # frequency for sinusoidal position term
    w2 = 3.0   # frequency for sinusoidal velocity term
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Basis functions
    b0 = np.tanh(x / s1)
    b1 = x3
    b2 = np.tanh(v / t1)
    b3 = v3
    b4 = x * v
    b5 = x2 * v
    b6 = v2 * x
    b7 = np.sin(w1 * x)
    b8 = np.cos(w2 * v)
    b9 = np.ones_like(x)
    
    # Combine with parameters
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
