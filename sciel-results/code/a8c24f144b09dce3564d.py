"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a hybrid basis expansion.
    
    Uses normalized inputs to ensure well-conditioned optimization.
    Basis:
    - Bias
    - Density (linear, quadratic)
    - Substrate (Monod-like saturation, normalized)
    - Temperature (linear, quadratic)
    - pH (linear, quadratic)
    - Interactions: Density*Temp, Substrate*Temp
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to approximately [-1, 1] for better numerical stability
    # b range: [1, 5] -> center 3, scale 2
    b_n = (b - 3.0) / 2.0
    
    # s range: [1, 5] -> Monod transform s/(1+s) range [0.5, 0.833]
    # Center ~0.667, Scale ~0.167
    s_monod = s / (1.0 + s)
    s_n = (s_monod - 0.667) / 0.167
    
    # temp range: [1, 40] -> center 20.5, scale 19.5
    T_n = (temp - 20.5) / 19.5
    
    # pH range: [2, 12] -> center 7, scale 5
    pH_n = (pH - 7.0) / 5.0

    # Construct prediction
    # p0: Bias
    pred = np.full_like(b_n, p0, dtype=np.float64)
    
    # p1, p2: Density terms
    pred += p1 * b_n
    pred += p2 * b_n * b_n
    
    # p3: Substrate term (Monod-like)
    pred += p3 * s_n
    
    # p4, p5: Temperature terms
    pred += p4 * T_n
    pred += p5 * T_n * T_n
    
    # p6, p7: pH terms
    pred += p6 * pH_n
    pred += p7 * pH_n * pH_n
    
    # p8: Interaction Density * Temperature
    pred += p8 * b_n * T_n
    
    # p9: Interaction Substrate * Temperature
    pred += p9 * s_n * T_n

    return pred
# EVOLVE-BLOCK-END
