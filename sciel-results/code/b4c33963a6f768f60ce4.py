# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    For each candidate, we draw a sample from the Gaussian posterior
    N(mu, sigma^2). We return the negative of this sample, so that the
    evaluator (which picks the maximum score) will select the candidate
    with the lowest predicted function value in this particular sample.
    
    This method naturally balances exploration and exploitation:
    - Exploitation: Candidates with low mu will tend to have low samples.
    - Exploration: Candidates with high sigma have a higher chance of
      producing an exceptionally low sample, encouraging the algorithm
      to visit uncertain regions.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a sample from the posterior for each candidate
    # Using a single random draw per candidate per call
    samples = mu + sigma_safe * np.random.standard_normal(mu.shape)
    
    # We want to minimize the function value.
    # The acquisition function should return scores where higher is better.
    # Therefore, we return the negative of the sampled values.
    score = -samples
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety check for time
    if time.time() > _DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
