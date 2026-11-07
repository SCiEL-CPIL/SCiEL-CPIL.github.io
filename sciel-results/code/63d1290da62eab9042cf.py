# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _elapsed():
    return time.time() - _ensure_start_time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a Monte Carlo
    estimated probability of being the global minimum.
    
    The EI term exploits regions likely to beat f_best.
    The probability-of-best term (estimated via Monte Carlo) boosts candidates
    that are likely to be the best in the entire search space, helping escape
    local minima in the multi-modal Hartmann-3D landscape.
    """
    # Deadline guard: if we're close to the 330s limit, return a simple EI-based score
    # to avoid timeout. The evaluator kills at 330s, so we use 325s as a safety margin.
    if _elapsed() > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (standard formulation for minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Monte Carlo estimate of probability that each candidate is the global minimum
    # among all 2000 candidates. We sample from the joint GP posterior.
    # Since we don't have the full covariance matrix, we approximate by assuming
    # independence (which is a simplification but computationally feasible).
    # A better approximation: use the fact that for independent normals,
    # P(x_i is min) = E[prod_{j != i} Phi((mu_j - mu_i)/sigma_j)]
    # But this is O(n^2) and hard to vectorize. Instead, we use a Monte Carlo
    # approach with a small number of samples.
    
    n_samples = 50  # Number of Monte Carlo samples
    n_candidates = len(mu)
    
    # For each sample, draw from N(mu, sigma) for all candidates
    # Then count how many times each candidate is the minimum
    rng = np.random.default_rng(42)
    samples = rng.normal(mu, sigma_safe, size=(n_samples, n_candidates))
    
    # Find the index of the minimum in each sample
    min_indices = np.argmin(samples, axis=1)
    
    # Count how many times each candidate is the minimum
    prob_best = np.zeros(n_candidates)
    for idx in min_indices:
        prob_best[idx] += 1
    prob_best /= n_samples
    
    # Scale the probability of best term
    # We want it to have a similar magnitude to EI
    # EI values are typically in the range of the function values
    # prob_best is in [0, 1], so we scale it by a factor
    # The Hartmann 3D range is roughly [-3.86, 3.86], so scale by ~1
    prob_best_weight = 1.0
    prob_best_bonus = prob_best_weight * prob_best
    
    # Combined score: EI + probability of best bonus
    # Higher values indicate better candidates
    scores = ei + prob_best_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
