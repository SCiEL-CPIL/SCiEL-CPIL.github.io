"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a sparse quadratic polynomial in normalized inputs.
    
    Strategy:
    1. Normalize inputs to [0, 1] based on known training ranges.
    2. Construct 10 monomial features: intercept, 4 linear, 4 quadratic, 2 key interactions.
    3. Output is a linear combination of these features with 10 learnable coefficients.
    
    Parameters (10 total):
    p[0]: Intercept
    p[1]: coeff for b_norm
    p[2]: coeff for s_norm
    p[3]: coeff for temp_norm
    p[4]: coeff for pH_norm
    p[5]: coeff for b_norm^2
    p[6]: coeff for s_norm^2
    p[7]: coeff for temp_norm^2
    p[8]: coeff for pH_norm^2
    p[9]: coeff for s_norm * temp_norm (key interaction)
    """
    # Normalization to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Compute monomial features
    f0 = np.ones_like(b_norm)
    f1 = b_norm
    f2 = s_norm
    f3 = temp_norm
    f4 = ph_norm
    f5 = b_norm * b_norm
    f6 = s_norm * s_norm
    f7 = temp_norm * temp_norm
    f8 = ph_norm * ph_norm
    f9 = s_norm * temp_norm
    
    # Linear combination
    prediction = (c0 * f0 + c1 * f1 + c2 * f2 + c3 * f3 + c4 * f4 + 
                  c5 * f5 + c6 * f6 + c7 * f7 + c8 * f8 + c9 * f9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(prediction)
        
    return prediction
# EVOLVE-BLOCK-END
