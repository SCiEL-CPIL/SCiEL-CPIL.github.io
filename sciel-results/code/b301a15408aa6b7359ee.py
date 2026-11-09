"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator (Duffing-like) with periodic drive.
    
    Model:
    a = -k1*x - k3*x^3 - c1*v - c3*v^3 + A_sin*sin(omega*t) + A_cos*cos(omega*t) + linear_t*t + bias
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p[0]: Linear spring constant (k1)
        p[1]: Cubic spring constant (k3)
        p[2]: Linear damping constant (c1)
        p[3]: Cubic damping constant (c3)
        p[4]: Sine drive amplitude
        p[5]: Cosine drive amplitude
        p[6]: Drive frequency (omega)
        p[7]: Linear time coefficient
        p[8]: Bias
        p[9]: Cross-term coefficient (x*v) for additional nonlinearity
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate non-linear features
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    
    # Calculate drive term with adjustable frequency
    # Using sin/cos with frequency p6 allows the model to fit the phase and frequency of the drive
    # If p6 is close to 1, this behaves like the previous model but with better phase alignment
    # If p6 differs, it captures frequency mismatch
    # Note: BFGS may struggle if p6 is far from 1, but start value is 1.0
    sin_drive = np.sin(p6 * t)
    cos_drive = np.cos(p6 * t)
    
    # Construct the acceleration
    # Signs are chosen to match standard oscillator notation (restoring/damping oppose motion)
    # However, since parameters are fitted, the sign is effectively absorbed into the parameter value.
    # We use + signs and let the optimizer find the correct sign for p0..p3.
    result = (
        p0 * x 
        + p1 * x3 
        + p2 * v 
        + p3 * v3 
        + p4 * sin_drive 
        + p5 * cos_drive 
        + p7 * t 
        + p8 
        + p9 * xv
    )
    
    return result
# EVOLVE-BLOCK-END
