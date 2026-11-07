"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization with deadline guard.
    # Score(x) = -mu(x) + kappa * sigma(x)
    # Higher score means evaluate next.
    
    # Deadline guard: ensure we return well before 330s wall-clock limit
    t_start = time.time()
    deadline = t_start + 280.0  # 50s safety margin
    
    # Kappa controls exploration-exploitation trade-off
    # For Ackley 2D with its multimodal landscape, moderate exploration is key
    kappa = 1.2
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Compute base LCB score
    scores = -mu + kappa * sigma
    
    # Check deadline after base computation
    if time.time() > deadline:
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Refine: dampen extreme uncertainty to focus on promising uncertain regions
    # Points with very high sigma relative to the range are likely in unexplored
    # or noisy regions; we want to explore but not waste evaluations on pure noise
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
        # Dampen scores where normalized sigma is very high (>0.95)
        # This keeps exploration but avoids pure noise regions
        damp_mask = sigma_norm > 0.95
        if np.any(damp_mask):
            # Apply a gentle penalty proportional to how far above threshold
            excess = (sigma_norm[damp_mask] - 0.95) / 0.05
            scores[damp_mask] -= 0.1 * excess * sigma_max
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
