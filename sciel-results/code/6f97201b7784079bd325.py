"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature using a bivariate Gaussian RBF network."""
    
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
    
    p = params
    
    # Parameters:
    # p[0]: bias (constant offset)
    # p[1], p[2]: center of RBF in (u, v) space
    # p[3], p[4]: widths (sigma) of RBF in (u, v) space
    # p[5]: amplitude (weight) of the RBF
    # p[6], p[7]: linear terms for u and v (background trend)
    # p[8], p[9]: quadratic terms for u and v (curvature)
    
    # Ensure widths are positive for numerical stability
    sigma_u = np.abs(p[3]) + 1e-6
    sigma_v = np.abs(p[4]) + 1e-6
    
    # Compute the Gaussian RBF term
    # exp(-((u - c_u)^2 / (2*sigma_u^2) + (v - c_v)^2 / (2*sigma_v^2)))
    du = (u - p[1]) / sigma_u
    dv = (v - p[2]) / sigma_v
    
    rbf = np.exp(-0.5 * (du * du + dv * dv))
    
    # Combine terms
    stress = (
        p[0] * 1.0 +
        p[5] * rbf +
        p[6] * u +
        p[7] * v +
        p[8] * (u * u) +
        p[9] * (v * v)
    )
    
    return stress
# EVOLVE-BLOCK-END
