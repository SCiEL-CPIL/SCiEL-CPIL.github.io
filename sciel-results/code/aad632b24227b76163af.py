# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Pure bivariate rational function (Padé-like) with polynomial numerator
    and denominator. This structural form naturally captures:
    - Linear elastic response (low-strain behavior)
    - Yield transition and plastic flow (inflection)
    - Saturation / asymptotic behavior (large strain)
    - Full temperature dependence in all coefficients
    
    Form: stress = N(s, t) / D(s, t)
    where:
      N = (p0 + p1*t) * s + (p2 + p3*t) * s^2 + p4 * s * t
      D = 1 + (p5 + p6*t) * s + (p7 + p8*t) * s^2 + p9 * s * t
    
    Inputs are used in raw form (no clipping) to preserve gradient information
    at the boundaries, which is important for the BFGS optimizer.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Use raw inputs for better gradient flow
    s = strain
    t = temp
    
    # Precompute powers
    s2 = s * s
    
    # Numerator: quadratic in s, linear in t, plus cross term
    num = (p0 + p1 * t) * s + (p2 + p3 * t) * s2 + p4 * s * t
    
    # Denominator: 1 + linear in s + quadratic in s, all with temp dependence, plus cross term
    den = 1.0 + (p5 + p6 * t) * s + (p7 + p8 * t) * s2 + p9 * s * t
    
    # Safe division: avoid division by zero or near-zero
    # Use a soft floor on the absolute value of the denominator
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8 * np.where(den >= 0, 1.0, -1.0), den)
    # Handle the exact-zero case
    den_safe = np.where(den_safe == 0.0, 1e-8, den_safe)
    
    stress = num / den_safe
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
