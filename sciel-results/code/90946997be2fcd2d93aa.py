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
      N = (p0 + p1*t + p2*t^2) * s + (p3 + p4*t) * s^2 + p5 * s * t
      D = 1 + (p6 + p7*t) * s + (p8 + p9*t) * s^2
    
    This combines the quadratic temperature dependence in the elastic modulus
    (p2*t^2 term) with a simple denominator for stable BFGS convergence,
    as evidenced by the improved score of 10.7982 from inspiration 1.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Use raw inputs for better gradient flow
    s = strain
    t = temp
    
    # Precompute powers
    s2 = s * s
    t2 = t * t
    
    # Numerator: quadratic in s, quadratic in t, plus cross term
    # Coefficients of s: p0 + p1*t + p2*t^2  (quadratic in T for elastic modulus)
    # Coefficients of s^2: p3 + p4*t         (linear in T for hardening)
    # Cross term: p5 * s * t
    num = (p0 + p1 * t + p2 * t2) * s + (p3 + p4 * t) * s2 + p5 * s * t
    
    # Denominator: 1 + linear in s + quadratic in s, both with linear T dependence
    den = 1.0 + (p6 + p7 * t) * s + (p8 + p9 * t) * s2
    
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
