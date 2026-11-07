"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible additive basis expansion.
    
    Uses:
    - Monod saturation for substrate
    - Gaussian temperature optimum
    - Gaussian pH optimum
    - Linear and quadratic density terms
    - Interaction terms between density and substrate, temperature, pH
    - Cross terms between temperature and pH
    
    This additive structure allows the optimizer to fit complex shapes
    without being constrained to a single mechanistic multiplicative form.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe parameters
    Ks = np.maximum(np.abs(p1), 1e-6)
    T_opt = p2
    T_width = np.maximum(np.abs(p3), 1e-3)
    pH_opt = p4
    pH_width = np.maximum(np.abs(p5), 1e-3)
    
    # Monod substrate term
    monod_s = s / (Ks + s)
    
    # Gaussian temperature factor
    temp_diff = temp - T_opt
    temp_factor = np.exp(-0.5 * (temp_diff / T_width) ** 2)
    
    # Gaussian pH factor
    ph_diff = pH - pH_opt
    ph_factor = np.exp(-0.5 * (ph_diff / pH_width) ** 2)
    
    # Density terms
    b_term = b
    b_sq_term = b ** 2
    
    # Base growth: linear combination of all basis functions
    # p0: intercept
    # p1 (already used for Ks, so we reuse the fitted value implicitly)
    # We use the remaining params for the linear coefficients
    
    # Basis functions:
    # 1. monod_s
    # 2. temp_factor
    # 3. ph_factor
    # 4. b_term
    # 5. b_sq_term
    # 6. monod_s * temp_factor
    # 7. monod_s * ph_factor
    # 8. temp_factor * ph_factor
    # 9. b_term * monod_s
    # 10. b_term * temp_factor
    
    result = (
        p0
        + p1 * monod_s
        + p2 * temp_factor
        + p3 * ph_factor
        + p4 * b_term
        + p5 * b_sq_term
        + p6 * monod_s * temp_factor
        + p7 * monod_s * ph_factor
        + p8 * temp_factor * ph_factor
        + p9 * b_term * monod_s
    )
    
    return np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
# EVOLVE-BLOCK-END
