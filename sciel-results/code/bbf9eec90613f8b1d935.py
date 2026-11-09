"""Structural change: additive exponential model with per-temperature hardening."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.

    Uses an additive model:
      sigma = (p0 + p1*strain + p2*strain^2) * exp(p3*temp + p4*strain*temp)
            + p5*strain*temp + p6*strain + p7*temp
            + p8 * tanh(p9 * strain)
    
    The exponential term captures the non-linear hardening/softening with
    strain and temperature. The tanh term adds a saturation effect at
    high strain. Linear and bilinear terms provide baseline flexibility.
    
    All 10 parameters are fitted.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base polynomial in strain
    base = p0 + p1 * strain + p2 * strain**2
    
    # Exponential modulation with temperature and strain interaction
    # Clip the exponent to avoid overflow
    exp_arg = p3 * temp + p4 * strain * temp
    exp_arg = np.clip(exp_arg, -30.0, 30.0)
    exp_term = base * np.exp(exp_arg)
    
    # Bilinear and linear correction terms
    corr = p5 * strain * temp + p6 * strain + p7 * temp
    
    # Saturation term using tanh
    tanh_arg = p9 * strain
    sat = p8 * np.tanh(tanh_arg)
    
    stress = exp_term + corr + sat
    
    return stress
# EVOLVE-BLOCK-END
