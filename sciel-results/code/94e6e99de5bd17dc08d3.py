# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a rational function with cubic strain terms and linear temperature terms
    in the numerator, and a linear denominator. 
    Scaled to initialize near the mean of the target data.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs
    # strain in [-0.001911, 0.9636] -> center 0.48, scale 0.48
    s = (strain - 0.48) / 0.48
    # temp in [0.06667, 1] -> center 0.53, scale 0.47
    t = (temp - 0.53) / 0.47
    
    # Numerator: Cubic in s, Quadratic in t, mixed terms. Scaled by 0.5
    num = 0.5 * (p0 + p1 * s + p2 * s**2 + p3 * s**3 + 
                 p4 * t + p5 * t**2 + 
                 p6 * s * t + p7 * s**2 * t)
                 
    # Denominator: Linear in s and t. Scaled by 0.1 to keep near 1.0
    den = 1.0 + 0.1 * (p8 * s + p9 * t)
    
    # Safety: ensure denominator is not zero
    # Since den is near 1, this is rarely triggered but good for safety
    den_safe = np.where(np.abs(den) < 1e-6, np.sign(den + 1e-300) * 1e-6, den)
    
    stress = num / den_safe
    
    return stress
# EVOLVE-BLOCK-END
