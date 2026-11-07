"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    LCB = mu - k * sigma
    
    For minimization, we want to find points where the expected value is low
    (low mu) and where there is high uncertainty (high sigma). The LCB balances
    these by subtracting a multiple of the standard deviation from the mean.
    
    A high exploration constant (k=3.0) is used to aggressively target regions
    with high posterior variance, which is effective for the Michalewicz function
    where the global minimum is isolated and surrounded by high-gradient ridges.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    t0 = time.time()
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    
    # Handle non-finite mu values
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Exploration constant for LCB
    k = 3.0
    
    # Compute LCB: lower confidence bound for minimization
    # We want to maximize this score (higher = more promising)
    # LCB = mu - k * sigma, but for minimization we want low values
    # So the acquisition score should be negative of LCB, or we can just
    # use -LCB = -mu + k*sigma, but that's not standard.
    
    # Actually, for minimization:
    # - We want low mu (exploit)
    # - We want high sigma (explore)
    # LCB for minimization: mu - k * sigma (lower is better)
    # Acquisition score: -LCB = -mu + k * sigma (higher is better)
    
    lcb = mu_safe - k * sigma_safe
    scores = -lcb  # Negate so higher score means more promising
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 5.0:
        pass  # Computation is already done, just return
    
    return scores
# EVOLVE-BLOCK-END
