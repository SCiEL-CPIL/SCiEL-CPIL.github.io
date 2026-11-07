"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a structured basis.
    
    Uses Chebyshev-like polynomial basis in x and v (degree 3),
    combined with sinusoidal time terms, to capture nonlinear dynamics.
    
    params: 10 fitted constants
    - p0: constant offset
    - p1: linear in x
    - p2: linear in v
    - p3: x^2
    - p4: v^2
    - p5: x*v
    - p6: x^3
    - p7: v^3
    - p8: sin(t)
    - p9: cos(t)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to roughly [-1, 1] for numerical stability
    t_scaled = t / 40.0  # t in [30, 50] -> roughly [0.75, 1.25]
    x_scaled = x / 0.2  # x in [-0.2056, 0.1749]
    v_scaled = v / 0.35  # v in [-0.3391, 0.3624]
    
    # Polynomial basis terms
    x2 = x_scaled ** 2
    x3 = x2 * x_scaled
    v2 = v_scaled ** 2
    v3 = v2 * v_scaled
    xv = x_scaled * v_scaled
    
    # Sinusoidal time terms
    sin_t = np.sin(2.0 * np.pi * t_scaled)
    cos_t = np.cos(2.0 * np.pi * t_scaled)
    
    result = (p0 + p1 * x_scaled + p2 * v_scaled + 
              p3 * x2 + p4 * v2 + p5 * xv +
              p6 * x3 + p7 * v3 +
              p8 * sin_t + p9 * cos_t)
    
    return result
# EVOLVE-BLOCK-END
