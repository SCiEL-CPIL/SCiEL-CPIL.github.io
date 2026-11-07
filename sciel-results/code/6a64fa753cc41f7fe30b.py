"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function using Monte Carlo approximation
    to estimate the probability of each candidate being the global minimum,
    combined with Expected Improvement.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        KG scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n_candidates = mu.shape[0]
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Monte Carlo estimation of the probability that each candidate is the
    # global minimum among the 2000 candidates.
    # We assume independence for the samples to allow vectorized computation.
    # This is an approximation but helps identify promising global basins.
    
    n_samples = 100  # Number of MC samples
    # Shape: (n_samples, n_candidates)
    # We sample from N(mu, sigma^2) for each candidate independently.
    # To save memory/time, we can do this in batches or just rely on numpy broadcasting.
    # Generating 100 * 2000 floats is small (200k floats).
    
    # Use a fixed seed for reproducibility in the MC step? 
    # The task says "fresh uniform candidates" are drawn by the evaluator, 
    # but the acquisition function is called with fixed mu/sigma.
    # We should use a deterministic or seeded RNG for the internal MC to ensure stability.
    rng = np.random.default_rng(42)
    
    # Generate samples: (n_samples, n_candidates)
    samples = rng.normal(loc=mu, scale=sigma_safe, size=(n_samples, n_candidates))
    
    # Find the minimum index in each sample
    min_indices = np.argmin(samples, axis=1)
    
    # Count how often each candidate is the minimum
    counts = np.bincount(min_indices, minlength=n_candidates)
    prob_global_min = counts / n_samples
    
    # Combine EI and Probability of Global Minimum
    # We want to boost candidates that are likely to be the global minimum.
    # A simple multiplicative boost:
    # Score = EI * (1 + alpha * prob_global_min)
    # Or additive:
    # Score = EI + beta * prob_global_min * (f_best - mu) ?
    
    # Let's use a weighted sum where the probability of being global min
    # scales the EI. Candidates with high EI and high prob of being global min
    # will be strongly favored.
    
    alpha = 1.0
    scores = ei * (1.0 + alpha * prob_global_min)
    
    return scores
# EVOLVE-BLOCK-END
