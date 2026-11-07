"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling for multimodal minimization.
    
    Samples from the posterior distribution N(mu, sigma^2) for each candidate.
    Returns the negative of the sample, so that the candidate with the lowest
    sampled function value receives the highest score. This stochastic approach
    allows the optimizer to probabilistically jump between different basins of
    attraction, which is beneficial for multimodal functions like Michalewicz 2D.
    """
    t_start = time.time()
    
    # Ensure proper types and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    
    # Check time budget early
    if time.time() - t_start > 280:
        # Fallback to exploitation if time is tight
        return -mu
    
    # Thompson Sampling:
    # Draw a sample from the posterior for each candidate.
    # y_i ~ N(mu_i, sigma_i)
    # Since the evaluator picks the candidate with the HIGHEST score,
    # and we want to minimize the function, we return -y_i.
    # This means the candidate with the most negative y_i (lowest function value)
    # will have the highest score (-y_i) and be selected.
    
    # Use standard normal random numbers
    z = np.random.standard_normal(n)
    
    # Sample values
    samples = mu + sigma * z
    
    # Return negative samples so that lower function values get higher scores
    score = -samples
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
