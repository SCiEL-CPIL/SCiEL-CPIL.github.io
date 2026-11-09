"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill


def _elapsed():
    return time.time() - _start_time


def _ei(mu, sigma, f_best):
    """Standard Gaussian Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)


def _knowledge_gradient(mu, sigma, f_best, n_samples=200):
    """
    Approximate Knowledge Gradient acquisition function.
    
    KG measures the expected reduction in the minimum objective value
    by considering the probability that each candidate becomes the new global best.
    
    We approximate this by:
    1. For each candidate i, compute P(f(x_i) < f_best) = Phi((f_best - mu_i) / sigma_i)
    2. The expected improvement of the minimum given that candidate i is evaluated
       is approximately: E[min(f_best, f(x_i))] reduction
    3. KG_i = sum over j of P(candidate j is best) * E[improvement | j is best]
    
    A simpler but effective approximation:
    KG_i ≈ E[max(0, f_best - f(x_i))] + exploration term based on how much
    evaluating x_i reduces uncertainty about the global minimum.
    
    The key insight: KG values points that are likely to be the global minimum,
    not just locally improving. We approximate this by:
    - The probability that x_i is better than all other candidates
    - Weighted by the expected improvement
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)
    
    # Probability that candidate i is better than f_best
    z = (f_best - mu) / sigma_safe
    p_better = norm.cdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * p_better + sigma_safe * norm.pdf(z)
    
    # For KG, we want to favor points that are likely to be the GLOBAL minimum.
    # A point is more valuable if it has a high probability of being the best
    # among all candidates. We approximate this by normalizing the "improvement potential."
    
    # The KG score for candidate i is approximately:
    # E[ min(f_best, f(x_i)) ] reduction, but weighted by how unique x_i is
    # as the global minimum.
    
    # Simple effective approximation:
    # - High EI points are good (exploitation)
    # - Points with high uncertainty that are near the current best are valuable
    #   because they could reveal a much better minimum
    # - We use a "global" view: normalize EI by the spread of EI values
    
    # Compute a global normalization to make scores comparable
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    # Add a term that rewards points with high probability of being the global min
    # This is approximated by: high p_better AND high sigma (uncertainty means
    # there could be a much better value hidden there)
    global_min_prob = p_better * (1.0 + 0.5 * sigma_safe / (np.max(sigma_safe) + 1e-12))
    
    # Combine: standard EI + global minimum probability bonus
    # The bonus encourages exploring uncertain regions that might contain the true minimum
    kg_score = ei + 0.3 * global_min_prob * ei_max
    
    return np.where(np.isfinite(kg_score), kg_score, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient-inspired acquisition function for Bayesian Optimization.
    
    Uses a combination of:
    - Expected Improvement (local exploitation)
    - Global minimum probability (KG-style global view)
    
    Parameters:
        mu: array of shape (N,), posterior mean in objective units
        sigma: array of shape (N,), posterior standard deviation in objective units
        f_best: scalar, lowest observed value so far
    
    Returns:
        scores: array of shape (N,), higher means more promising to evaluate next
    """
    # Deadline guard: if we're close to timeout, return simple EI quickly
    if _elapsed() > _DEADLINE_SECONDS:
        return _ei(mu, sigma, f_best)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Knowledge Gradient approximation
    kg_scores = _knowledge_gradient(mu, sigma, f_best)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(kg_scores), kg_scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
