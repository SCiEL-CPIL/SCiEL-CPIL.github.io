import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod-like saturation for substrate
    and Fourier series for temperature and pH to capture optimal ranges.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to assist optimization of frequency/phase parameters
    # temp range approx [1, 40] -> center 20, scale 20
    t_norm = (temp - 20.0) / 20.0
    # pH range approx [2, 12] -> center 7, scale 5
    ph_norm = (pH - 7.0) / 5.0
    
    # Substrate effect: Saturating exponential
    # p1 is the scale factor. We add epsilon to prevent division by zero if p1->0
    # Using max(p1, 1e-6) to ensure positive scale for the exponent
    p1_safe = np.maximum(p1, 1e-6)
    substrate_term = p0 * (1.0 - np.exp(-s / p1_safe))
    
    # Temperature effect: Fourier basis
    # A single sine wave can approximate a bell curve over a limited range
    temp_term = p2 * np.sin(p3 * t_norm + p4)
    
    # pH effect: Fourier basis
    ph_term = p5 * np.sin(p6 * ph_norm + p7)
    
    # Population density effect: Linear
    # High density often inhibits growth, so p8 might be negative
    density_term = p8 * b
    
    # Global bias
    intercept = p9
    
    return substrate_term + temp_term + ph_term + density_term + intercept
