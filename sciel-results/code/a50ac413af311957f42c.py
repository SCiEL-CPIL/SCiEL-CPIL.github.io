"""Combined program: rational Padé-like form with cross-terms and quadratic temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines the rational Padé-like structure of the current program with
    cross-term and quadratic-temperature refinements from the inspirations.
    
    Form:
      stress = N(s, t) / D(s, t)
      where
        N = (p0 + p1*t + p2*t^2) * s + (p3 + p4*t) * s^2 + p5 * s * t
        D = 1 + (p6 + p7*t) * s + (p8 + p9*t) * s^2
    
    This captures:
      - Linear elastic response at low strain
      - Yield transition and plastic flow via rational inflection
      - Saturation / asymptotic behavior at large strain
      - Quadratic temperature softening typical of aluminium
      - Cross-coupling between strain and temperature
    """
    s = np.asarray(strain, dtype=np.float64)
    t = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
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
    abs_den = np.abs(den)
    den_safe = np.where(abs_den < 1e-8, 1e-8 * np.where(den >= 0, 1.0, -1.0), den)
    # Handle the exact-zero case
    den_safe = np.where(den_safe == 0.0, 1e-8, den_safe)
    
    stress = num / den_safe
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
