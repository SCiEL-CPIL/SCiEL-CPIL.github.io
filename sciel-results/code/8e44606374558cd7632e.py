"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a Fourier-like basis with a linear term
    # This can capture periodic and non-periodic smooth functions
    # Domain [-3, 3]
    
    # Frequency scaling: map [-3, 3] to [-pi, pi] for standard harmonics
    # omega = pi / 3
    omega = np.pi / 3.0
    
    # Basis functions
    # p0: DC term
    # p1, p2: 1st harmonic (sin, cos)
    # p3, p4: 2nd harmonic
    # p5, p6: 3rd harmonic
    # p7, p8: 4th harmonic
    # p9: linear term x (to handle non-periodic drift)
    
    sin1 = np.sin(omega * x)
    cos1 = np.cos(omega * x)
    
    sin2 = np.sin(2.0 * omega * x)
    cos2 = np.cos(2.0 * omega * x)
    
    sin3 = np.sin(3.0 * omega * x)
    cos3 = np.cos(3.0 * omega * x)
    
    sin4 = np.sin(4.0 * omega * x)
    cos4 = np.cos(4.0 * omega * x)
    
    y = (params[0] + 
         params[1] * sin1 + 
         params[2] * cos1 + 
         params[3] * sin2 + 
         params[4] * cos2 + 
         params[5] * sin3 + 
         params[6] * cos3 + 
         params[7] * sin4 + 
         params[8] * cos4 + 
         params[9] * x)
         
    return y
# EVOLVE-BLOCK-END
