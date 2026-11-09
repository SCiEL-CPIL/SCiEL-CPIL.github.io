"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    This method samples a value from the posterior distribution N(mu, sigma^2)
    for each candidate point. The candidate with the lowest sampled value is
    selected for the next evaluation. This approach naturally balances
    exploration and exploitation without requiring explicit tuning of
    exploration-exploitation trade-off parameters like beta in UCB or
    improvement thresholds in EI.
    
    Args:
        mu: numpy array of shape (2000,) containing posterior means.
        sigma: numpy array of shape (2000,) containing posterior standard deviations.
        f_best: float, the lowest observed function value so far.
        
    Returns:
        numpy array of shape (2000,) with scores. Higher scores indicate
        candidates that are more likely to be the global minimum.
    """
    # Ensure sigma is strictly positive to handle numerical edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw one sample from the posterior for each candidate
    # samples[i] ~ N(mu[i], sigma[i]^2)
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # We are minimizing the function.
    # The evaluator selects the candidate with the MAXIMUM score.
    # Therefore, we assign scores such that lower sampled values correspond
    # to higher scores.
    scores = -samples
    
    # Safety check: ensure all scores are finite
    # If any score is not finite (e.g., due to extreme values), replace with -inf
    # so it won't be selected.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
