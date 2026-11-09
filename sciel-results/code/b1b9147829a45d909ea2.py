"""Pareto-based multi-objective acquisition for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pareto-based multi-objective acquisition function.
    
    Combines normalized Expected Improvement (EI) and normalized predictive uncertainty
    with a dynamic weight that increases with the number of evaluations (proxied by
    the spread of sigma relative to the scale of the problem), balancing exploitation
    and exploration.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std dev in objective units
        f_best: float - best observed value so far (lowest)
    
    Returns:
        scores: array of shape (2000,) - higher scores mean evaluate next
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute predictive uncertainty (use sigma as a proxy for uncertainty)
    # Normalize uncertainty to [0, 1] range using a soft scaling
    # The Hartmann 6D has a range of roughly 3.3 units, so we scale accordingly
    # Use a dynamic scale based on the observed sigma distribution
    sigma_scale = np.median(sigma) + 1e-9
    uncertainty = sigma / sigma_scale
    
    # Cap uncertainty to avoid extreme values
    uncertainty = np.clip(uncertainty, 0.0, 10.0)
    
    # Dynamic weight: starts with more exploration, shifts to exploitation
    # As optimization progresses, f_best approaches the minimum, EI becomes smaller
    # We use the ratio of best EI to typical EI to gauge progress
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1e-9
    ei_normalized = ei / ei_max
    
    # The weight on exploration decreases as we find better solutions
    # Use a heuristic: when EI values are large relative to the problem scale,
    # we are early in optimization; when small, we are late
    # Problem scale for Hartmann 6D is ~3.3
    problem_scale = 3.32237  # minimum value magnitude
    
    # Progress indicator: how close is f_best to the known minimum?
    # We don't know the true minimum, but we can use the spread of mu
    # Early on, mu has high variance; later, it's more concentrated
    mu_spread = np.std(mu)
    # Normalize spread to [0, 1] using a logistic-like function
    # When mu_spread is large, we're exploring; when small, exploiting
    progress = 1.0 / (1.0 + np.exp(-3.0 * (mu_spread / problem_scale)))
    
    # Weight on exploitation (EI) increases with progress
    # Weight on exploration (uncertainty) decreases with progress
    w_exploit = 0.3 + 0.7 * (1.0 - progress)  # 0.3 to 1.0
    w_explore = 1.0 - w_exploit  # 1.0 to 0.3
    
    # Combine the two objectives
    score = w_exploit * ei_normalized + w_explore * uncertainty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
