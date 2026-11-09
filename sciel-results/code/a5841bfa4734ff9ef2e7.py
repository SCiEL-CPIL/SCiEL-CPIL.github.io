"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition function.
    
    For each candidate, we draw K samples from the posterior Gaussian N(mu, sigma^2)
    and return the negative of the minimum of those samples. This approximates an
    Upper Confidence Bound (UCB) strategy with a stochastic component, balancing
    exploration and exploitation more effectively than single-sample TS in
    low-budget regimes.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at each candidate point.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Number of samples to draw per candidate.
    # K=10 approximates a UCB with beta ~ 2.15
    K = 10
    
    # Draw K samples from each posterior Gaussian N(mu_i, sigma_i^2)
    # Shape: (2000, K)
    Z = np.random.standard_normal((len(mu), K))
    samples = mu[:, np.newaxis] + sigma[:, np.newaxis] * Z
    
    # Take the minimum of the K samples for each candidate
    # This favors candidates with lower means and penalizes high uncertainty
    min_samples = np.min(samples, axis=1)
    
    # We want to minimize the objective, so higher acquisition score = lower sampled value
    # Return negative samples so that argmax(scores) corresponds to argmin(samples)
    return -min_samples
# EVOLVE-BLOCK-END
