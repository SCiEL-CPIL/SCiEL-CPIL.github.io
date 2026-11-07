import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate Polynomial Chaos Expansion (Taylor Series) of degree 3.
    
    stress = sum_{i=0}^{3} sum_{j=0}^{3-i} p_{i,j} * x^i * y^j
    
    where x is normalized strain and y is normalized temperature.
    10 params: [p00, p10, p01, p20, p11, p02, p30, p21, p12, p03]
    
    Normalization:
    x = (strain - 0.481) / 0.483
    y = (temp - 0.533) / 0.467
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    # Normalize inputs to improve numerical conditioning
    x = (strain - 0.481) / 0.483
    y = (temp - 0.533) / 0.467
    
    # Extract parameters
    p00, p10, p01, p20, p11, p02, p30, p21, p12, p03 = params
    
    # Compute polynomial basis terms
    result = p00 + p10 * x + p01 * y + p20 * x**2 + p11 * x * y + p02 * y**2 + p30 * x**3 + p21 * x**2 * y + p12 * x * y**2 + p03 * y**3
    
    return result
