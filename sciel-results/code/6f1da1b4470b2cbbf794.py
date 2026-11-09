"""Symbolic regression with non-linear composite parametric model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear composite model with 10 parameters.
    
    Model:
    y = c0 + c1*x + c2*x^2 + c3*x^3 + 
        A * exp(-((x - mu)^2) / (2 * sigma^2)) + 
        B * sin(omega * x + phi)
        
    Parameters:
    0: c0 (polynomial constant)
    1: c1 (polynomial linear)
    2: c2 (polynomial quadratic)
    3: c3 (polynomial cubic)
    4: A  (Gaussian amplitude)
    5: mu (Gaussian center)
    6: sigma_raw (Gaussian width, transformed to positive)
    7: B  (Sine amplitude)
    8: omega (Sine frequency)
    9: phi (Sine phase)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    A = params[4]
    mu = params[5]
    sigma_raw = params[6]
    B = params[7]
    omega = params[8]
    phi = params[9]
    
    # Polynomial part
    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    # Gaussian part
    # Ensure sigma is positive and not too small to avoid division by zero or overflow
    # Use softplus with clipping for numerical stability
    sigma = np.log1p(np.exp(np.clip(sigma_raw, -50, 50))) + 0.1
    # Clip the exponent argument to avoid overflow/underflow
    gauss_arg = -((x - mu)**2) / (2.0 * sigma**2)
    gauss_arg = np.clip(gauss_arg, -500, 0)
    gauss = A * np.exp(gauss_arg)
    
    # Sine part
    # Clip omega to avoid extreme frequencies causing numerical issues
    omega_clipped = np.clip(omega, -20, 20)
    sine = B * np.sin(omega_clipped * x + phi)
    
    # Combine
    y = poly + gauss + sine
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
