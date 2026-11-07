"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a flexible 4th-order multivariate polynomial.
    
    This approach abandons the assumption of independent factorization (Monod/Gaussian)
    and allows the optimizer to fit arbitrary high-order interaction terms and
    non-monotonic dependencies across all variables simultaneously.
    
    Inputs are normalized to [0, 1] range based on observed training data ranges:
    b: [1, 5] -> [0, 1]
    s: [1, 5] -> [0, 1]
    temp: [1, 40] -> [0, 1]
    pH: [2, 12] -> [0, 1]
    
    params (length 10):
    p[0]: Constant term
    p[1]: Coefficient for b
    p[2]: Coefficient for s
    p[3]: Coefficient for temp
    p[4]: Coefficient for pH
    p[5]: Coefficient for b*s
    p[6]: Coefficient for temp*pH
    p[7]: Coefficient for b*temp
    p[8]: Coefficient for s*pH
    p[9]: Coefficient for 4th-order interaction (b*s*temp*pH)
    """
    p = params
    
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to avoid extrapolation issues if inputs are slightly out of range
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Construct polynomial terms
    # We select a sparse set of 10 terms that cover main effects, key interactions, and a high-order term
    # This allows for complex non-linearities and interactions without overfitting with 7500 points
    
    term_0 = np.ones_like(b_norm)
    term_1 = b_norm
    term_2 = s_norm
    term_3 = temp_norm
    term_4 = ph_norm
    
    # Quadratic terms for curvature (non-monotonic behavior)
    term_5 = b_norm * s_norm
    term_6 = temp_norm * ph_norm
    term_7 = b_norm * temp_norm
    term_8 = s_norm * ph_norm
    
    # High-order interaction to capture complex dependencies
    term_9 = b_norm * s_norm * temp_norm * ph_norm
    
    # Combine terms
    growth = (p[0] * term_0 + 
              p[1] * term_1 + 
              p[2] * term_2 + 
              p[3] * term_3 + 
              p[4] * term_4 + 
              p[5] * term_5 + 
              p[6] * term_6 + 
              p[7] * term_7 + 
              p[8] * term_8 + 
              p[9] * term_9)
    
    return growth
# EVOLVE-BLOCK-END
