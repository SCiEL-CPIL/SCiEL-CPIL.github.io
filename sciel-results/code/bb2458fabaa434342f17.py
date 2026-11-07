import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a bivariate Taylor series (polynomial chaos) of total degree 3.
    This provides a smooth, differentiable surface with 10 coefficients.
    """
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_w = s_max - s_min
    t_w = t_max - t_min
    
    # Normalize to [-1, 1] for better conditioning
    u = 2.0 * (strain - s_min) / s_w - 1.0
    v = 2.0 * (temp - t_min) / t_w - 1.0
    
    # Compute powers
    u2 = u * u
    u3 = u2 * u
    v2 = v * v
    v3 = v2 * v
    
    # Bivariate Taylor series terms of total degree <= 3
    # Order:
    # 0: 1
    # 1: u
    # 2: v
    # 3: u^2
    # 4: u*v
    # 5: v^2
    # 6: u^3
    # 7: u^2*v
    # 8: u*v^2
    # 9: v^3
    
    term_00 = np.ones_like(u)
    term_10 = u
    term_01 = v
    term_20 = u2
    term_11 = u * v
    term_02 = v2
    term_30 = u3
    term_21 = u2 * v
    term_12 = u * v2
    term_03 = v3
    
    return (params[0] * term_00 +
            params[1] * term_10 +
            params[2] * term_01 +
            params[3] * term_20 +
            params[4] * term_11 +
            params[5] * term_02 +
            params[6] * term_30 +
            params[7] * term_21 +
            params[8] * term_12 +
            params[9] * term_03)
