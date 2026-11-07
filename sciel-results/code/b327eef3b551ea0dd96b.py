"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a single sample from the GP posterior for each candidate point.
    The score is defined as the negative of the sampled value divided by the
    evaluation cost. This encourages selecting points with low predicted
    function values and low costs, while naturally balancing exploration and
    exploitation through the stochastic sampling.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each point (1 + mean(x) for x in [0,1]^6).
    
    Returns:
    --------
    scores : array_like, shape (2000,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero in sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive (it should be >= 1, but just in case)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate standard normal noise for Thompson Sampling
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    
    # Draw samples from the posterior: f(x) ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # Cost-aware score: minimize (sample / cost)
    # Equivalent to maximizing -(sample / cost)
    # Using a power of cost can tune the sensitivity. The directive suggests
    # "divided by its cost", so we use cost ** 1.0.
    # However, sometimes cost ** 0.5 or similar is used to soften.
    # Let's stick to the directive's literal interpretation first.
    scores = - (samples / cost_safe)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
