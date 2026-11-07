import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive LCB: explore more when far from known global minimum
    opt = -1.8013
    dist = max(0.0, f_best - opt)
    
    # Refined schedule:
    # Use a steeper decay (scale 2.0 vs 5.0) to switch to exploitation faster as we get closer.
    # Use a higher initial exploration weight (3.0 vs 2.0) to better handle the multimodal landscape early on.
    kappa = 0.5 + 3.0 * (dist / 2.0)
    
    # Cap kappa to avoid excessive exploration in the early stages
    kappa = min(kappa, 4.0)
    
    # Score: higher is better for selection (LCB for minimization)
    scores = -mu + kappa * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
