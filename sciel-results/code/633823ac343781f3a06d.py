# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Multiplicative interaction model with power-law saturation:
    sigma = (p0 + p1*s) * (p2 + p3*t) * (1 + p4*s^p5) / (1 + p6*s^p7 + p8*t^p9) + p10
    
    This decouples:
    - Linear elastic scaling in strain: (p0 + p1*s)
    - Temperature-dependent stiffness: (p2 + p3*t)
    - Non-linear strain saturation via power law: (1 + p4*s^p5)
    - Denominator provides additional saturation and temperature coupling
    - p10 is an offset
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize strain to [0, 1]
    s = (strain + 0.002) / 0.966
    s = np.clip(s, 0.0, 1.0)
    
    # Normalize temperature to [0, 1]
    t = (temp - 0.06667) / 0.93333
    t = np.clip(t, 0.0, 1.0)
    
    # Ensure exponents are positive to avoid issues with 0^negative
    p5_safe = np.abs(p5) + 0.1
    p7_safe = np.abs(p7) + 0.1
    p9_safe = np.abs(p9) + 0.1
    
    # Compute power terms safely
    s_p5 = np.power(s, p5_safe)
    s_p7 = np.power(s, p7_safe)
    t_p9 = np.power(t, p9_safe)
    
    # Multiplicative model
    elastic = p0 + p1 * s
    temp_factor = p2 + p3 * t
    saturation_num = 1.0 + p4 * s_p5
    den = 1.0 + p6 * s_p7 + p8 * t_p9
    
    # Safe division
    den_safe = np.where(np.abs(den) < 1e-12, 1e-12 * np.sign(den), den)
    den_safe = np.where(den_safe == 0, 1e-12, den_safe)
    
    stress = elastic * temp_factor * saturation_num / den_safe + p9
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
