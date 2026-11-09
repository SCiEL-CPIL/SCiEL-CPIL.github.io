import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Start time for deadline guard (though not strictly needed for O(N) ops)
    start_time = time.time()
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Compute Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # We choose beta to balance exploration. 
    # A common choice is beta=1, but we can scale it.
    # Let's use a beta that is proportional to the scale of the problem.
    # Since mu and sigma are in objective units, beta=1 is a reasonable start.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Ensure UCB is finite
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # 3. Normalize EI and UCB to [0, 1] range
    # This makes the combination scale-invariant
    
    def min_max_normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    ei_norm = min_max_normalize(ei)
    ucb_norm = min_max_normalize(ucb)
    
    # 4. Combine scores
    # Weight 0.5 for exploitation (EI) and 0.5 for exploration (UCB)
    # This is a structural change from pure EI.
    scores = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Deadline check (should be instantaneous)
    if time.time() - start_time > 290:
        # Fallback to simple EI if somehow slow
        return ei
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
