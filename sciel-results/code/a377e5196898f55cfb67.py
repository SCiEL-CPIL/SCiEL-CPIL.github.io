import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are too close to the limit, return a safe fallback
    # Since this function is very fast, this is unlikely to trigger, but it's required.
    if time.time() > _DEADLINE:
        # Fallback to a simple EI-like score to avoid hanging
        return -mu + 0.1 * sigma

    # Keep the Expected Improvement (EI) design and investigate improvements
    # within its family. Here we use a temperature-scaled EI to adjust the
    # exploration/exploitation tradeoff. As uncertainty shrinks in later
    # iterations, the raw EI can become too greedy; dividing by a temperature
    # that grows mildly with the number of evaluations (proxied by how
    # close mu is to f_best) smooths the score and can lead to better
    # global coverage of the 2D RBF mixture.
    #
    # We compute the standard EI:
    #   z = (f_best - mu) / sigma
    #   EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #
    # Then we return EI / temperature, where temperature is derived from the
    # median sigma. This keeps the EI family but adds a smooth, data-driven
    # scaling that acts like a soft-min on the improvement landscape.
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Temperature scaling: larger temperature when sigma is large (early),
    # smaller when sigma is small (late). This is the inverse of typical
    # temperature annealing but here it normalizes EI so that early high-
    # uncertainty regions are not overemphasized.
    med_sigma = np.median(sigma_safe)
    # Clip to avoid extreme values
    temp = np.clip(med_sigma, 1e-6, 1.0)
    
    # Return temperature-scaled EI
    scores = ei / temp
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
