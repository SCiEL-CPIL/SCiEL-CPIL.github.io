import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial chaos expansion with Legendre polynomials.
    Degree 4 in strain, degree 1 in temperature.
    Orthogonal basis improves numerical conditioning for the optimizer.
    """
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_w = s_max - s_min
    t_w = t_max - t_min
    
    s_n = (strain - s_min) / s_w
    t_n = (temp - t_min) / t_w
    
    # Map to [-1, 1] for Legendre polynomials
    u = 2.0 * s_n - 1.0
    v = 2.0 * t_n - 1.0
    
    # Compute Legendre polynomials for strain (u)
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    L0 = np.ones_like(u)
    L1 = u
    L2 = 0.5 * (3.0 * u2 - 1.0)
    L3 = 0.5 * (5.0 * u3 - 3.0 * u)
    L4 = 0.125 * (35.0 * u4 - 30.0 * u2 + 3.0)
    
    # Compute Legendre polynomials for temperature (v)
    M0 = np.ones_like(v)
    M1 = v
    
    # Combine basis terms with parameters
    # Basis: L_i(u) * M_j(v)
    # i=0..4, j=0..1
    # Order: (L0M0, L1M0, L2M0, L3M0, L4M0, L0M1, L1M1, L2M1, L3M1, L4M1)
    
    return (params[0] * L0 * M0 +
            params[1] * L1 * M0 +
            params[2] * L2 * M0 +
            params[3] * L3 * M0 +
            params[4] * L4 * M0 +
            params[5] * L0 * M1 +
            params[6] * L1 * M1 +
            params[7] * L2 * M1 +
            params[8] * L3 * M1 +
            params[9] * L4 * M1)
