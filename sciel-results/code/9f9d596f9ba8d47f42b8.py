"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and Expected Simple Regret (ESR)
    with cost-awareness. The cost-aware term penalizes expensive evaluations.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        GP mean predictions for N candidate points.
    sigma : ndarray, shape (N,)
        GP standard deviation predictions for N candidate points.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (N,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores (higher is better) for each candidate.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Expected Simple Regret: E[max(f_best - f(x), 0)]
    # This is the same as EI in this formulation, but let's also consider
    # the probability of improvement for exploration
    prob_improve = norm.cdf(z)
    
    # Cost-aware scoring: we want high improvement per unit cost
    # Use a soft cost-awareness that balances exploitation and cost
    # score = EI / cost would be too aggressive; use a blend
    
    # Primary: cost-aware EI
    score_ei = ei / np.maximum(cost, 1e-6)
    
    # Secondary: exploration bonus based on uncertainty, also cost-aware
    # Higher sigma means more exploration value
    score_explore = sigma_safe / np.maximum(cost, 1e-6)
    
    # Tertiary: probability of improvement, cost-aware
    score_prob = prob_improve / np.maximum(cost, 1e-6)
    
    # Combine: weight EI most heavily, with some exploration and probability
    # The weights can be tuned; start with these and see
    score = 1.0 * score_ei + 0.3 * score_explore + 0.2 * score_prob
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative (scores should be interpretable)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
