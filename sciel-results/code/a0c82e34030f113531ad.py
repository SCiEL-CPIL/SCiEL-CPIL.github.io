"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative power-law saturation model.
    
    Model:
    sigma = E(T) * eps * (1 + A(T) * eps)^(-n(T)) + thermal_correction
    
    This captures:
    - Linear elastic behavior at small strain: sigma ~ E(T)*eps
    - Power-law saturation at large strain (strain hardening -> plateau)
    - Temperature modulates modulus, hardening amplitude, and saturation exponent
    - Small thermal stress correction
    
    Parameters (10 constants):
    p0, p1, p2: Elastic modulus E(T) = p0 + p1*T + p2*T^2
    p3, p4: Hardening amplitude A(T) = p3 + p4*T  (must keep 1+A*eps > 0)
    p5, p6: Saturation exponent n(T) = p5 + p6*T
    p7: Thermal stress coefficient (linear in T)
    p8: Linear thermal correction (T * eps)
    p9: Quadratic strain correction
    p10: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T          # Elastic modulus
    A_T = p3 + p4 * T                         # Hardening amplitude
    n_T = p5 + p6 * T                         # Saturation exponent
    
    # Ensure the base (1 + A*eps) stays positive for numerical stability
    # Use a soft floor: if 1+A*eps would go negative, clamp it
    base = 1.0 + A_T * eps
    # For safety, ensure base > 0 (use max with small epsilon)
    base_safe = np.maximum(base, 1e-8)
    
    # Power-law saturation term
    # At eps->0: (1+A*eps)^(-n) ~ 1, so sigma ~ E*eps (linear elastic)
    # At eps->inf: sigma ~ E * eps * (A*eps)^(-n) = E/A^n * eps^(1-n)
    # If n > 1, this saturates to a plateau
    saturation = np.power(base_safe, -n_T)
    
    main_term = E_T * eps * saturation
    
    # Thermal and correction terms
    thermal_term = p7 * T + p8 * T * eps
    quad_term = p9 * eps * eps
    
    sigma = main_term + thermal_term + quad_term
    
    return sigma
# EVOLVE-BLOCK-END
