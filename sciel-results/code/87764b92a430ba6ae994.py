"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the session
_SESSION_START = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard
    elapsed = time.time() - _SESSION_START
    if elapsed > 290:
        sigma_safe = np.maximum(sigma, 1e-8)
        # Fallback: simple UCB/cost
        score = (-mu + 1.5 * sigma_safe) / np.maximum(cost, 1e-8)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample from the posterior for each candidate
    # For minimization, we want to find points with low values
    # Sample: y_i = mu_i + sigma_i * z_i where z_i ~ N(0,1)
    # We want to maximize the "improvement" score: (f_best - y_i) / cost_i
    # But since we're sampling, we use the negative of the sample (lower = better)
    # Score = -sample / cost, so higher score means more promising
    
    # Generate random samples
    rng = np.random.default_rng()
    z = rng.standard_normal(mu.shape)
    
    # Sampled function values (lower is better)
    sampled_values = mu + sigma_safe * z
    
    # Score: we want to maximize the potential improvement
    # Use (f_best - sampled_value) as the improvement, normalized by cost
    # This gives higher scores to points likely to beat f_best, weighted by cost
    improvement = f_best - sampled_values
    
    # Cost-normalized score
    score = improvement / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
