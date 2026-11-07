"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a rich basis combining:
    - Orthonormal Legendre polynomials (up to degree 9)
    - Sinusoidal terms with learnable frequencies
    - Cosine terms
    - Gaussian bumps
    This allows BFGS to fit a wider class of hidden functions.
    
    The 10 params are used as:
    - params[0:5] : coefficients for 5 orthonormal Legendre polynomials
    - params[5:7] : frequencies for sin and cos terms
    - params[7:9] : amplitudes for sin and cos terms
    - params[9]   : width parameter for a Gaussian bump centered at 0
    """
    t0 = time.time()
    
    x = np.asarray(x, dtype=float).flatten()
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Normalize input to [-1, 1]
    t = x / 3.0
    
    # Compute orthonormal Legendre polynomials up to degree 4 (5 terms)
    # P_0 = 1, orthonormal: 1
    # P_1 = t, orthonormal: sqrt(3)*t
    # P_2 = (3t^2 - 1)/2, orthonormal: sqrt(5)*(3t^2-1)/2
    # P_3 = (5t^3 - 3t)/2, orthonormal: sqrt(7)*(5t^3-3t)/2
    # P_4 = (35t^4 - 30t^2 + 3)/8, orthonormal: sqrt(9)*(35t^4-30t^2+3)/8
    
    P0 = np.ones_like(t)
    P1 = t
    P2 = (3.0 * t**2 - 1.0) / 2.0
    P3 = (5.0 * t**3 - 3.0 * t) / 2.0
    P4 = (35.0 * t**4 - 30.0 * t**2 + 3.0) / 8.0
    
    # Orthonormal scaling
    leg0 = P0  # sqrt(1) = 1
    leg1 = np.sqrt(3.0) * P1
    leg2 = np.sqrt(5.0) * P2
    leg3 = np.sqrt(7.0) * P3
    leg4 = 3.0 * P4  # sqrt(9) = 3
    
    # Polynomial part: 5 terms
    poly_part = (params[0] * leg0 + 
                 params[1] * leg1 + 
                 params[2] * leg2 + 
                 params[3] * leg3 + 
                 params[4] * leg4)
    
    # Sinusoidal terms with learnable frequencies
    freq_sin = params[5]
    freq_cos = params[6]
    amp_sin = params[7]
    amp_cos = params[8]
    
    # Safety: clamp frequencies to avoid numerical issues
    freq_sin = np.clip(freq_sin, -20.0, 20.0)
    freq_cos = np.clip(freq_cos, -20.0, 20.0)
    
    sin_part = amp_sin * np.sin(freq_sin * t)
    cos_part = amp_cos * np.cos(freq_cos * t)
    
    # Gaussian bump centered at 0
    width = params[9]
    # Ensure width is positive and reasonable
    width = np.clip(width, 0.1, 10.0)
    gauss_part = np.exp(-0.5 * (t / width)**2)
    
    # Combine all terms
    result = poly_part + sin_part + cos_part + gauss_part
    
    # Deadline guard: if we're running long, ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
