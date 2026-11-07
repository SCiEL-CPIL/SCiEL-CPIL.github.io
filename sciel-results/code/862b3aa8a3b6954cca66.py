"""Thompson Sampling acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling with adaptive exploitation bias for Michalewicz 2D.
    
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
    
    # Adaptive exploitation bias:
    # 1. Points with mu < f_best get a boost proportional to how much they beat f_best
    # 2. Points with high uncertainty (large sigma) relative to their potential get a smaller boost
    #    to encourage exploration in uncertain regions
    # 3. Use a soft max to smoothly blend exploitation and exploration
    
    # Potential improvement: how much could this point beat f_best?
    potential_improvement = np.maximum(f_best - mu, 0)
    
    # Uncertainty-adjusted exploitation:
    # If sigma is large, we're less confident, so reduce exploitation bias
    # This prevents over-exploitation of noisy regions
    uncertainty_penalty = sigma_safe / (np.abs(f_best) + 1.0)
    adjusted_potential = potential_improvement / (1.0 + uncertainty_penalty)
    
    # Combine Thompson sample with exploitation bias
    # Higher score = more likely to be evaluated next
    # -ts_samples: lower objective values get higher scores
    # adjusted_potential: points that can beat f_best get extra credit
    exploitation_weight = 0.5
    score = -ts_samples + exploitation_weight * adjusted_potential
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
