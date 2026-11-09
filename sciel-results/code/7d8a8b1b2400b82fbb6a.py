"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Fourier series on [-2, 2] with period 4.
    
    y(x) = a0 + a1*cos(w*x) + b1*sin(w*x)
             + a2*cos(2w*x) + b2*sin(2w*x)
             + a3*cos(3w*x) + b3*sin(3w*x)
             + a4*cos(4w*x) + b4*sin(4w*x)
             + a5*cos(5w*x)
    
    where w = pi/2 (fundamental angular frequency for period 4).
    
    Parameters (10 total):
    p[0]  = a0  (DC / constant term)
    p[1]  = a1  (1st harmonic cosine)
    p[2]  = b1  (1st harmonic sine)
    p[3]  = a2  (2nd harmonic cosine)
    p[4]  = b2  (2nd harmonic sine)
    p[5]  = a3  (3rd harmonic cosine)
    p[6]  = b3  (3rd harmonic sine)
    p[7]  = a4  (4th harmonic cosine)
    p[8]  = b4  (4th harmonic sine)
    p[9]  = a5  (5th harmonic cosine)
    
    The trigonometric basis is orthogonal on [-2, 2], yielding well-conditioned
    gradients for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Time guard: ensure we finish well within the 150s limit
    # (evaluation is fast, but guard for safety)
    
    # Fundamental angular frequency for period 4 on [-2, 2]
    w = np.pi / 2.0
    
    a0 = p[0]
    a1, b1 = p[1], p[2]
    a2, b2 = p[3], p[4]
    a3, b3 = p[5], p[6]
    a4, b4 = p[7], p[8]
    a5 = p[9]
    
    # Compute basis functions
    wx = w * x
    c1 = np.cos(wx)
    s1 = np.sin(wx)
    c2 = np.cos(2.0 * wx)
    s2 = np.sin(2.0 * wx)
    c3 = np.cos(3.0 * wx)
    s3 = np.sin(3.0 * wx)
    c4 = np.cos(4.0 * wx)
    s4 = np.sin(4.0 * wx)
    c5 = np.cos(5.0 * wx)
    
    # Combine: linear combination of basis functions
    result = (a0 
              + a1 * c1 + b1 * s1 
              + a2 * c2 + b2 * s2 
              + a3 * c3 + b3 * s3 
              + a4 * c4 + b4 * s4 
              + a5 * c5)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
