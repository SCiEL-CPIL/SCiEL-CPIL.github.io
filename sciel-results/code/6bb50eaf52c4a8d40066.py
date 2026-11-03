"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to maximize the minimum radius
    by minimizing the sum of squared inverse distances.
    """
    n = N
    
    # 1. Generate Initial Guess: Phyllotaxis (Sunflower) Pattern
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(1, n + 1)
    r_max = 0.95
    r = r_max * np.sqrt(i / n)
    theta = i * golden_angle
    x_init = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    x0 = x_init.flatten()
    
    def residuals(x):
        """
        Minimize sum of squared inverse distances.
        """
        centres = x.reshape(n, 2)
        
        # Compute full distance matrix
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Extract upper triangle
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        # Add epsilon
        pair_dists = pair_dists + 1e-8
        
        # Boundary distances
        norms = np.linalg.norm(centres, axis=1)
        bdists = 1.0 - norms
        bdists = bdists + 1e-8
        
        # Residuals
        pair_res = 1.0 / pair_dists
        bound_res = 1.0 / bdists
        
        return np.concatenate([pair_res, bound_res])
    
    # Run optimization
    try:
        result = least_squares(residuals, x0, method='trf', max_nfev=500, 
                               xtol=1e-6, ftol=1e-6, gtol=1e-6)
        centres = result.x.reshape(n, 2)
    except Exception:
        centres = x_init

    return centres
# EVOLVE-BLOCK-END
