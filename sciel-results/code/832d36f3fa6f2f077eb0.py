"""
Bayesian Optimization Acquisition Function for Hartmann 6d
Strategy: Adaptive LCB with a smooth surrogate that balances exploration and
exploitation, tuned for the 6D Hartmann function with 30 evaluations.
"""
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses an adaptive Lower Confidence Bound (LCB) with a smooth surrogate to
    balance exploration and exploitation, tuned for the 6D Hartmann function
    with 30 evaluations.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before the 330s kill limit

    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Adaptive LCB: use a high exploration coefficient to target regions of
    # high posterior uncertainty, which is critical for escaping local optima
    # in the multimodal Hartmann 6D landscape within a tight 30-evaluation budget.
    kappa = 3.0

    # Compute the base LCB score: -mu + kappa * sigma
    # Higher score = more promising (lower expected value, higher uncertainty)
    base_scores = -mu + kappa * sigma_safe

    # Apply a smooth surrogate to the base scores to create a more robust
    # ranking that is less sensitive to outliers in the posterior.
    # Use a soft-max-like transformation to smooth the distribution of scores.
    # This helps in balancing exploration and exploitation more effectively.
    
    # First, normalize the base scores to [0, 1] range for better numerical stability
    min_base = np.min(base_scores)
    max_base = np.max(base_scores)
    range_base = max_base - min_base
    
    if range_base > 1e-12:
        normalized_scores = (base_scores - min_base) / range_base
    else:
        normalized_scores = np.zeros_like(base_scores)

    # Apply a smooth transformation that emphasizes the top scores while
    # maintaining a smooth gradient for optimization
    # Use a scaled logistic function to create a smooth ranking
    temperature = 0.1  # Controls the sharpness of the smooth transformation
    smooth_scores = np.log(1.0 + np.exp(normalized_scores / temperature))

    # Combine the smooth scores with the original base scores to maintain
    # both the smooth ranking and the original LCB signal
    # Weight the smooth scores heavily to ensure robustness
    alpha = 0.7
    scores = alpha * smooth_scores + (1 - alpha) * base_scores

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    # Check deadline
    if time.time() - start_time > deadline:
        return scores

    return scores
