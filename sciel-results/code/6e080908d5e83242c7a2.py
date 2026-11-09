import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290
_SEED_COUNTER = 0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        return 1.0 / (1.0 + cost)
    
    global _SEED_COUNTER
    _SEED_COUNTER += 1
    
    # Clip sigma for numerical stability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Thompson Sampling: Sample from posterior of improvement
    # Improvement = f_best - f(x) ~ N(f_best - mu, sigma^2)
    # We want to maximize expected improvement, so we sample a realization
    # of the improvement and use that as our score.
    
    # Create a random generator with a deterministic seed based on call count
    # This ensures reproducibility if needed, but different seeds per call
    # to get different samples each time
    rng = np.random.default_rng(_SEED_COUNTER)
    
    # Sample from the improvement distribution
    # improvement ~ N(f_best - mu, sigma^2)
    sampled_improvement = rng.normal(f_best - mu, sigma)
    
    # Cost-aware scoring: divide by cost to penalize expensive evaluations
    scores = sampled_improvement / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
