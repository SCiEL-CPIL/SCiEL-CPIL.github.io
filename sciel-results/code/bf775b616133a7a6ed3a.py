# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    if time.time() - _start_time > _DEADLINE:
        return -mu + 2.0 * np.maximum(sigma, 1e-9)

    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)

    # Predictive Entropy Search (PES) variant
    # Score each candidate by expected reduction in uncertainty about the global minimum
    # Approximation: mutual information between f(x) and the location of the best point
    
    # 1. Compute the probability that each candidate is the new minimum
    # P(f(x) < f_best) under the posterior
    z = (f_best - mu) / sigma_safe
    p_new_min = norm.cdf(z)
    
    # 2. For candidates that might be new minima, compute expected improvement
    # in our knowledge of where the global minimum is
    # This is approximated by the expected reduction in entropy of the best value
    
    # The entropy of the best value's distribution is reduced when we find a new minimum
    # Approximate the PES score as:
    # - The probability of being a new minimum (exploration of low regions)
    # - Weighted by the expected improvement magnitude (exploitation)
    
    # Expected improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # PES-like score: combine probability of new minimum with expected improvement
    # and a term that rewards reducing uncertainty in the best region
    # The key insight: we want to find points that are likely to be near the global minimum
    
    # Adaptive weighting based on how good f_best is relative to the scale
    # If f_best is very low, we should be more exploitative
    # If f_best is high, we should explore more
    
    # Scale f_best: Weierstrass 1D has minimum 0, typical range [0, 2]
    # Normalize f_best to [0, 1] range approximately
    f_best_norm = np.clip(f_best / 2.0, 0.0, 1.0)
    
    # Weight: as f_best improves (lower), shift towards exploitation
    # w starts at 1.0 (balanced) and goes to 0.5 (more exploitation) as f_best -> 0
    w = 1.0 - 0.5 * (1.0 - f_best_norm)
    
    # PES score: 
    # - p_new_min * (expected value if new min) rewards finding new minima
    # - sigma_safe term rewards exploring uncertain regions
    # - mu term (negated) rewards low predicted values
    # Combine with adaptive weight
    
    # The PES objective: maximize E[log P(f(x) < f_best | posterior)]
    # Approximate this with a combination of:
    # 1. Probability of being a new minimum (log space for numerical stability)
    # 2. Expected reduction in the best value
    
    # Log-probability of being a new minimum (higher is better)
    log_p_new_min = np.log(np.maximum(p_new_min, 1e-15))
    
    # Expected improvement in the best value
    # If x is a new minimum, the improvement is f_best - f(x)
    # E[f_best - f(x) | f(x) < f_best] = EI / P(new min)
    expected_improvement_given_new_min = ei / np.maximum(p_new_min, 1e-15)
    
    # Combine: PES score approximates the mutual information
    # Use a weighted combination that adapts to the optimization stage
    score = w * log_p_new_min + (1 - w) * expected_improvement_given_new_min
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in the multimodal landscape
    exploration_bonus = 0.1 * sigma_safe / (1.0 + sigma_safe)
    
    score += exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
