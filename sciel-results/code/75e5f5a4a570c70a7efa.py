"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Use a Fourier-like basis with learnable frequencies and amplitudes.
    This targets periodic or quasi-periodic functions which polynomial bases struggle with.
    
    Model:
      y = p0 + p1*x + p2*sin(p3*x) + p4*cos(p5*x) + p6*sin(p7*x) + p8*cos(p9*x)
      
    Parameters:
      p0: constant offset
      p1: linear coefficient
      p2: amplitude for sin term 1
      p3: frequency for sin term 1
      p4: amplitude for cos term 1
      p5: frequency for cos term 1
      p6: amplitude for sin term 2
      p7: frequency for sin term 2
      p8: amplitude for cos term 2
      p9: frequency for cos term 2
    
    This representation allows the model to capture oscillatory behavior with adjustable
    frequencies, which is a different structural family from the previous polynomial+activation mix.
    """
    p = params
    
    # Clip frequencies to a reasonable range to avoid high-frequency oscillations
    # that BFGS might struggle with, but wide enough to capture various periods.
    # Period T = 2*pi/f. If f=0.1, T~62. If f=10, T~0.6.
    # Inputs are in [-2, 2], so max "cycles" for f=10 is ~ 40/6.28 ~ 6 cycles.
    # Let's allow frequencies from -20 to 20.
    
    f1 = np.clip(p[3], -20.0, 20.0)
    f2 = np.clip(p[5], -20.0, 20.0)
    f3 = np.clip(p[7], -20.0, 20.0)
    f4 = np.clip(p[9], -20.0, 20.0)
    
    # Amplitudes and linear/constant terms are unbounded (but BFGS will keep them sane)
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[4]
    a4 = p[6]
    a5 = p[8]
    
    # Compute terms
    term1 = a2 * np.sin(f1 * x)
    term2 = a3 * np.cos(f2 * x)
    term3 = a4 * np.sin(f3 * x)
    term4 = a5 * np.cos(f4 * x)
    
    result = a0 + a1 * x + term1 + term2 + term3 + term4
    
    # Ensure finite output
    return np.where(np.isfinite(result), result, 0.0)
# EVOLVE-BLOCK-END
