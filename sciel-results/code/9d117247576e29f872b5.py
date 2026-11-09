# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Structural change: pure rational function form.
    Uses a bivariate rational expression with polynomial numerator and
    denominator to capture the full stress-strain-temperature relationship
    without explicit elastic/plastic branching. The rational form naturally
    captures saturation behavior and smooth transitions.
    
    Form: stress = N(s, t) / D(s, t)
    where N and D are low-degree polynomials in normalized strain s and
    temperature t, with 10 total coefficients.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [0, 1] range for better conditioning
    s = (strain + 0.002) / 0.966
    t = (temp - 0.06667) / 0.93333
    
    # Numerator: quadratic in s, linear in t
    # N = (p0 + p1*t) * s + (p2 + p3*t) * s^2 + p4 * s * t
    num = (p0 + p1 * t) * s + (p2 + p3 * t) * s**2 + p4 * s * t
    
    # Denominator: 1 + (p5 + p6*t) * s + (p7 + p8*t) * s^2 + p9 * s * t
    den = 1.0 + (p5 + p6 * t) * s + (p7 + p8 * t) * s**2 + p9 * s * t
    
    # Safe division
    den_safe = np.where(np.abs(den) < 1e-10, 1e-10 * np.sign(den), den)
    den_safe = np.where(den_safe == 0, 1e-10, den_safe)
    
    stress = num / den_safe
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
