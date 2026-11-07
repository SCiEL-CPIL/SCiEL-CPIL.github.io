"""Thompson Sampling acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling variant with exploitation bias.
    
    Args:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior standard deviation in objective units
        f_best: float - lowest observed value so far
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    # Ensure sigma is positive and has a reasonable floor to maintain exploration
    # A floor of 1e-2 prevents the algorithm from becoming purely greedy in well-explored regions
    sigma_safe = np.maximum(sigma, 1e-2)
    
    # Thompson Sampling: sample from the posterior predictive
    # We want to minimize, so we sample mu - sigma * z where z ~ N(0,1)
    # This gives a random draw from the predictive distribution at each point
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    ts_samples = mu - sigma_safe * z
    
    # Add a exploitation bias toward known best
    # Increased weight to 0.5 to accelerate convergence to low values
    exploitation_weight = 0.5
    exploitation_bias = exploitation_weight * np.maximum(f_best - mu, 0)
    
    # Combine: Thompson sample minus exploitation bias (since lower is better for objective,
    # but we want higher score to mean "evaluate next")
    # ts_samples are in objective space (lower is better)
    # We want higher score = more likely to be evaluated
    # So score = -ts_samples + exploitation_bias
    # The -ts_samples makes lower objective values get higher scores
    # The exploitation_bias adds extra score to points that beat f_best
    
    score = -ts_samples + exploitation_bias
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
