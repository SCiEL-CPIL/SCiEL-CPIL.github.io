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
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from the posterior predictive
    # We want to minimize, so we sample mu - sigma * z where z ~ N(0,1)
    # This gives a random draw from the predictive distribution at each point
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    ts_samples = mu - sigma_safe * z
    
    # Add a small exploitation bias toward known best
    # The idea: slightly boost points that are near f_best to encourage exploitation
    # while the Thompson sample provides exploration
    # Use a small deterministic term that favors lower mu values
    
    # Exploitation bias: subtract a small multiple of (f_best - mu) when mu < f_best
    # This gives extra credit to points that beat the current best
    exploitation_weight = 0.1
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
