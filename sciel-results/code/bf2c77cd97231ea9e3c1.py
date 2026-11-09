"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are close to the limit, return a safe fallback.
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 295:
        z = (f_best - mu) / (sigma + 1e-9)
        return norm.cdf(z)
    
    # Use a Quantile-based acquisition function that maximizes the predicted
    # 5th percentile of the posterior distribution.
    # This directly targets the lower tail of the uncertainty, providing a more
    # robust and aggressive exploitation signal.
    
    # The 5th percentile is mu - 1.645 * sigma
    # We want to MINIMIZE this value (lower is better for the objective)
    # So we negate it to create a maximization score
    quantile_5 = mu - 1.645 * sigma
    scores = -quantile_5
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
