"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_ACQ_START_TIME = None
_ACQ_DEADLINE = 330.0  # Wall-clock limit with margin


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) inspired acquisition with sublinear cost discount.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions for 2000 candidates.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidates.
    f_best : float
        Current best observed value (lowest).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > _ACQ_DEADLINE - 5:
        # Safety fallback: return EI/cost quickly
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of being a new minimum
    prob_new_min = norm.cdf(z)
    
    # Knowledge Gradient approximation:
    # The KG value for candidate x is the expected improvement in the global optimum.
    # For a single candidate, KG ≈ EI(x) * P(x is the new minimum among all candidates)
    # But we can't compute the full KG exactly for 2000 points efficiently.
    # Instead, we use a surrogate: weight EI by a term that captures the likelihood
    # of this point being the best improvement.
    
    # A practical KG surrogate: use the probability of improvement, but also
    # consider the relative ranking. Points with higher EI and higher probability
    # of being the new min are more valuable.
    
    # Sublinear cost discount: 1/sqrt(cost)
    cost_discount = 1.0 / np.sqrt(cost_safe)
    
    # Primary term: EI with sublinear cost discount
    primary = ei * cost_discount
    
    # Secondary term: bonus for high probability of being new minimum
    # This encourages exploring regions where improvement is likely
    exploration_bonus = 0.15 * prob_new_min * sigma_safe * cost_discount
    
    # Tertiary term: mild penalty for very high cost to favor cheaper evaluations
    # when EI is comparable
    cost_penalty = 0.03 * (cost_safe - 1.0) ** 2
    
    # Final score
    scores = primary + exploration_bonus - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
