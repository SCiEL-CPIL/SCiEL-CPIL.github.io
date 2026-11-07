"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function with cost awareness.
    
    Strategy:
    1. Draw a single sample from the posterior for each candidate:
       s_i = mu_i + sigma_i * z_i, where z_i ~ N(0, 1)
    2. Score candidates by -s_i / cost_i.
    3. Return the top-1 score to all candidates (or a masked version) to 
       encourage diversity and avoid local clustering. However, the interface
       requires returning 2000 scores. The evaluator picks the max. 
       
       Wait, the evaluator picks the argmax of the returned scores. 
       If I return the same score for all, it picks the first one (index 0), which is arbitrary.
       I must return distinct scores such that the best one is the one I want.
       
       Actually, standard Thompson Sampling returns the sampled value (or negative for minimization).
       The evaluator will pick the candidate with the highest score.
       So I should return -sampled_value / cost.
       
       Why is this different from EI?
       EI is deterministic. TS is stochastic.
       TS naturally explores because high sigma leads to high variance in the sample,
       occasionally yielding very low (good) samples even in unexplored regions.
       
       Also, the prompt mentions "diversity-promoting strategy".
       TS is known to be more exploratory than EI in early stages.
       
       To further promote diversity, I can add a small noise term to the scores
       or use a "kernel" based penalty if we had coordinates, but we don't.
       However, the GP's sigma already encodes spatial information (high sigma = far from data).
       So TS on (mu, sigma) is inherently spatially aware.
       
       The cost normalization is preserved.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes and safety
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard
    if time.time() - _START > _LIMIT:
        # Fallback to exploitation
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Thompson Sampling
    # Generate standard normal noise
    # Using a fixed seed? No, stochastic is the point.
    # But wait, if the evaluator runs this multiple times, we want consistent behavior?
    # No, the evaluator runs the loop once. Each call to acquisition is one step.
    # The stochasticity is across calls.
    
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # We want to minimize, so lower samples are better.
    # Score = -sample / cost
    scores = -samples / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
