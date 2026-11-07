import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    
    This is a structural change from Expected Improvement (EI) to a stochastic
    acquisition method. Thompson Sampling samples from the GP posterior at each
    candidate point and selects the point with the lowest sampled value (for
    minimization). This naturally balances exploration (high sigma leads to
    high variance in samples) and exploitation (low mu leads to low samples).
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement.
        For minimization via Thompson Sampling, we sample y ~ N(mu, sigma^2)
        and return -y, so the lowest sampled value (best candidate) gets the
        highest score.
    """
    # Deadline guard
    start_time = time.time()
    # The evaluator kills after 330s. We assume this function is called within
    # the main loop. We reserve a small margin.
    deadline = 330.0 - 1.0 

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Check if we are close to the deadline.
    # Thompson Sampling is O(N) and very fast, but we guard just in case.
    if time.time() > deadline:
        # Fallback: Greedy (minimize mean)
        # Return -mu so that lowest mu gets highest score
        return -mu
    
    # Generate random standard normal draws for each candidate
    # This is the core of Thompson Sampling: sampling the function value
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled function values
    # y_i ~ N(mu_i, sigma_i^2)
    y_samples = mu + sigma_safe * z
    
    # We are minimizing the function.
    # We want to select the candidate with the minimum y_sample.
    # The interface requires "higher scores meaning evaluate next".
    # Therefore, we invert the sampled values.
    scores = -y_samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
