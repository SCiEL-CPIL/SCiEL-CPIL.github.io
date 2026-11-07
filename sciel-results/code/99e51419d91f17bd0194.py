"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier Basis approach
    # Uses 10 parameters to fit a Fourier series with 5 harmonics
    # Domain is [-3, 3], so fundamental frequency omega = pi / 3
    
    omega = np.pi / 3.0
    
    # Precompute basis functions for efficiency
    # We can compute sin and cos for each harmonic
    # y = p0 + p1*cos(w*x) + p2*sin(w*x) + ...
    
    # To be efficient and avoid recomputing sin/cos for every parameter index in a loop,
    # we can compute the necessary angles.
    
    # However, since we have only 10 params, a simple direct calculation is fast enough.
    # Let's structure it clearly.
    
    # params[0]: DC term
    result = np.full_like(x, params[0], dtype=float)
    
    # Harmonic 1
    angle1 = omega * x
    result += params[1] * np.cos(angle1) + params[2] * np.sin(angle1)
    
    # Harmonic 2
    angle2 = 2.0 * omega * x
    result += params[3] * np.cos(angle2) + params[4] * np.sin(angle2)
    
    # Harmonic 3
    angle3 = 3.0 * omega * x
    result += params[5] * np.cos(angle3) + params[6] * np.sin(angle3)
    
    # Harmonic 4
    angle4 = 4.0 * omega * x
    result += params[7] * np.cos(angle4) + params[8] * np.sin(angle4)
    
    # Harmonic 5 (Cosine only to use 10th param)
    angle5 = 5.0 * omega * x
    result += params[9] * np.cos(angle5)
    
    return result
# EVOLVE-BLOCK-END
