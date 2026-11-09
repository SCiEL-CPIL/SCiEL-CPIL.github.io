"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses Monte Carlo samples from the GP posterior to score candidates by the
    expected minimum cost-normalized objective value across samples. This
    inherently balances exploration and exploitation without heuristic blending.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline - if we're running out of time, return a simple score
    if time.time() > _DEADLINE - 5.0:
        # Simple fallback: negative mu / cost
        cost_safe = np.maximum(cost, 1e-10)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Number of Thompson samples - balance between accuracy and speed
    n_samples = 20
    
    # Ensure sigma is safe for sampling
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # For Thompson sampling, we want to find the candidate that minimizes
    # the expected cost-normalized objective across posterior samples.
    # 
    # For each sample j, we draw y_j(x) ~ N(mu(x), sigma(x)^2)
    # We score candidate x by: -mean_j [ y_j(x) / cost(x) ]
    # Higher score = better (lower expected cost-normalized value)
    
    # Generate random samples efficiently
    # Shape: (n_samples, 2000)
    rng = np.random.default_rng()
    eps = rng.standard_normal((n_samples, len(mu)))
    
    # Sample values: y_j = mu + sigma * eps_j
    # Shape: (n_samples, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * eps
    
    # Cost-normalized samples: y_j / cost
    # Shape: (n_samples, 2000)
    cost_norm_samples = samples / cost_safe[None, :]
    
    # For Thompson sampling, we want to select the candidate that has the
    # lowest expected cost-normalized value. But we also want to account for
    # the fact that we want to find the global minimum.
    #
    # A better approach: for each candidate, compute the expected minimum
    # cost-normalized value across samples, then negate for a higher-is-better score.
    #
    # Actually, the standard Thompson sampling approach is:
    # 1. Draw a single sample from the posterior for each candidate
    # 2. Select the candidate with the best (lowest) sampled value
    #
    # But since we need to return scores for all 2000 candidates, we use:
    # score(x) = -E_j [ y_j(x) / cost(x) ]
    #
    # This is the expected negative cost-normalized value, which balances
    # exploration (high sigma leads to high variance in samples) and
    # exploitation (low mu leads to low expected value).
    
    # Expected cost-normalized value for each candidate
    expected_cost_norm = np.mean(cost_norm_samples, axis=0)
    
    # Score: higher is better, so negate
    scores = -expected_cost_norm
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local minima in the multimodal Rastrigin landscape
    # The bonus is proportional to sigma / cost, weighted by how far f_best is from 0
    if f_best > 1.0:
        explore_weight = min(0.15, (f_best - 1.0) / 20.0)
        explore_term = explore_weight * sigma_safe / cost_safe
        scores = scores + explore_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
