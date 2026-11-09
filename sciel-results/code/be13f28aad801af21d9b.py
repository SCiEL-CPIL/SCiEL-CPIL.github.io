"""E. coli growth rate model: pure linear model in precomputed features."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model as a pure linear combination of 10 precomputed features.
    
    This structure makes the objective quadratic in params, so BFGS converges reliably.
    Features include:
    - Monod saturation for substrate
    - Gaussian responses for temperature and pH
    - Polynomial terms for density
    - Interaction terms
    - Bias
    """
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Unpack 10 parameters as linear coefficients
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Precompute features (all independent of params)
    # Monod term
    monod = s_norm / (0.5 + s_norm + 1e-9)
    
    # Temperature Gaussian (fixed center at 0.5, width 0.3)
    temp_term = np.exp(-((temp_norm - 0.5) ** 2) / (2.0 * 0.09 + 1e-9))
    
    # pH Gaussian (fixed center at 0.45, width 0.2)
    ph_term = np.exp(-((ph_norm - 0.45) ** 2) / (2.0 * 0.04 + 1e-9))
    
    # Density polynomial terms
    b_lin = b_norm
    b_quad = b_norm ** 2
    
    # Interaction terms
    monod_temp = monod * temp_term
    monod_ph = monod * ph_term
    temp_ph = temp_term * ph_term
    monod_b = monod * b_norm
    
    # Linear combination of 10 features
    result = (c0 * monod + 
              c1 * temp_term + 
              c2 * ph_term + 
              c3 * b_lin + 
              c4 * b_quad + 
              c5 * monod_temp + 
              c6 * monod_ph + 
              c7 * temp_ph + 
              c8 * monod_b + 
              c9)
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
