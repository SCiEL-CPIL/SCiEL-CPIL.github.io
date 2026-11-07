import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a bivariate polynomial chaos expansion based on Legendre polynomials.
    This provides an orthogonal basis which improves numerical stability and optimization conditioning.
    """
    s = strain
    t = temp
    
    # Normalize inputs to roughly [-1, 1] to improve scaling
    # Strain range approx [-0.002, 0.964], center ~ 0.48, scale ~ 0.48
    # Temp range approx [0.067, 1.0], center ~ 0.53, scale ~ 0.47
    s_n = (s - 0.48) / 0.48
    t_n = (t - 0.53) / 0.47
    
    # Precompute powers of normalized variables
    s_n2 = s_n * s_n
    s_n3 = s_n2 * s_n
    t_n2 = t_n * t_n
    t_n3 = t_n2 * t_n
    
    # Legendre polynomials P0..P3
    # P0(x) = 1
    # P1(x) = x
    # P2(x) = 1.5 x^2 - 0.5
    # P3(x) = 2.5 x^3 - 1.5 x
    
    L0_s = 1.0
    L1_s = s_n
    L2_s = 1.5 * s_n2 - 0.5
    L3_s = 2.5 * s_n3 - 1.5 * s_n
    
    L0_t = 1.0
    L1_t = t_n
    L2_t = 1.5 * t_n2 - 0.5
    L3_t = 2.5 * t_n3 - 1.5 * t_n
    
    # Construct the expansion: Sum of params[i] * Basis[i]
    # Order:
    # 0: L0_s * L0_t
    # 1: L1_s * L0_t
    # 2: L0_s * L1_t
    # 3: L2_s * L0_t
    # 4: L1_s * L1_t
    # 5: L0_s * L2_t
    # 6: L3_s * L0_t
    # 7: L2_s * L1_t
    # 8: L1_s * L2_t
    # 9: L0_s * L3_t
    
    return (params[0] * L0_s * L0_t +
            params[1] * L1_s * L0_t +
            params[2] * L0_s * L1_t +
            params[3] * L2_s * L0_t +
            params[4] * L1_s * L1_t +
            params[5] * L0_s * L2_t +
            params[6] * L3_s * L0_t +
            params[7] * L2_s * L1_t +
            params[8] * L1_s * L2_t +
            params[9] * L0_s * L3_t)
