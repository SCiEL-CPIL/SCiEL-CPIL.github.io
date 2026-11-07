"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature using a bivariate cubic polynomial."""
    
    # Normalization constants derived from the training data ranges provided in the task contract
    e_min = -0.001911
    e_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    e_range = e_max - e_min
    t_range = t_max - t_min
    
    # Ensure ranges are non-zero
    if e_range < 1e-12:
        e_range = 1.0
    if t_range < 1e-12:
        t_range = 1.0
        
    # Normalize inputs to [0, 1]
    u = (strain - e_min) / e_range
    v = (temp - t_min) / t_range
    
    # Clip to [0, 1] to ensure stability if test points are slightly out of bounds
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Pre-compute powers
    u2 = u * u
    u3 = u2 * u
    v2 = v * v
    v3 = v2 * v
    
    # Basis functions for a total degree 3 polynomial in u and v
    # There are 10 terms:
    # 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3
    
    p = params
    
    # Calculate the linear combination
    # Using explicit terms for clarity and speed
    stress = (
        p[0] * 1.0 +
        p[1] * u +
        p[2] * v +
        p[3] * u2 +
        p[4] * (u * v) +
        p[5] * v2 +
        p[6] * u3 +
        p[7] * (u2 * v) +
        p[8] * (u * v2) +
        p[9] * v3
    )
    
    return stress
# EVOLVE-BLOCK-END
