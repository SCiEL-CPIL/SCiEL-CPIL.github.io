# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Samples from the GP posterior and penalizes by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Finite acquisition scores (higher = more desirable).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a fixed seed for reproducibility in the sampling process
    # Note: In a real loop, this would use the global RNG or a passed seed,
    # but for this static function, a fixed seed ensures deterministic behavior
    # for the same inputs, which is safer for evaluation.
    rng = np.random.default_rng(42)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # z ~ N(0, 1)
    z = rng.standard_normal(mu.shape)
    y_samples = mu + sigma * z
    
    # We want to minimize y_samples.
    # The score should be high for low y_samples.
    # Also, we want to penalize high cost.
    # Score = -y_samples / cost
    # This favors points with low predicted value and low cost.
    
    score = -y_samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(score), score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
