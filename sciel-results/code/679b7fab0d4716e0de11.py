"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0


def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1

    # Deadline guard: if we are too close to the limit, return a simple score
    # to ensure we don't timeout.
    if time.time() - _start_time > 325:
        # Fallback to a simple greedy score
        return -mu

    # Use Deterministic Thompson Sampling
    # Seed the random generator with the call count to ensure reproducibility
    # while maintaining the stochastic nature of Thompson Sampling.
    # This helps balance exploration and exploitation in small budgets.
    rng = np.random.RandomState(_call_count)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # We want to minimize the objective, so we return -sample
    # so that the evaluator (picking max score) selects the point
    # with the lowest sampled value.
    samples = rng.normal(mu, sigma)
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
