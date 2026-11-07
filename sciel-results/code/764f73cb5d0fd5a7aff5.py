"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    We sample y_i ~ N(mu_i, sigma_i) for each candidate and select the one with
    the lowest sampled value. To convert this to a score where higher is better,
    we return -y_i.
    
    Since the evaluator calls this function once per step and selects the argmax,
    and since the randomness in Thompson Sampling is meant to be used for selection,
    we must be careful. The standard Thompson Sampling procedure is:
    1. Sample y_i for all i.
    2. Select i* = argmin y_i.
    
    However, the interface requires returning scores such that the evaluator selects argmax.
    If we simply return -y_i, the evaluator will select argmax(-y_i) = argmin(y_i), which is correct.
    
    But wait: the evaluator calls `acquisition` and then picks the candidate with the highest score.
    If we use a single random draw, the result is stochastic. This is the intended behavior of TS.
    
    To ensure numerical stability and avoid issues with sigma being zero, we add a small epsilon.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Generate standard normal random numbers
    # Use a fixed seed? No, TS requires randomness.
    # However, if the evaluator expects deterministic behavior for a given mu/sigma/f_best,
    # TS is inherently stochastic. The task says "higher meaning evaluate next".
    # Let's check if there's any constraint against randomness.
    # "Return 2000 finite scores".
    # Stochasticity is fine as long as scores are finite.
    
    # Draw samples
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    
    # We want the evaluator to pick the minimum y_sample.
    # So we return -y_samples.
    scores = -y_samples
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
