import numpy as np
import time

_START = time.time()
_DEADLINE = 150.0 - 10.0

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod substrate kinetics
    monod = s / (p1 + s)
    
    # Temperature bell (Gaussian)
    temp_bell = np.exp(-((temp - p2) / np.maximum(np.abs(p3), 0.1)) ** 2)
    
    # pH bell (Gaussian)
    ph_bell = np.exp(-((pH - p4) / np.maximum(np.abs(p5), 0.1)) ** 2)
    
    # Density inhibition (linear)
    density = 1.0 - p7 * b
    
    # Main term
    result = p0 * monod * temp_bell * ph_bell * density
    
    # Additive corrections using remaining params
    result = result + p6 + p8 * monod * (temp - p2) + p9 * monod * (pH - p4)
    
    return result
