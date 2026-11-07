import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function using Monte Carlo integration.
    
    The 1-step Knowledge Gradient is equivalent to Expected Improvement (EI):
    KG(x) = E[max(0, f_best - f(x))]
    
    We approximate this expectation by sampling from the Gaussian posterior
    N(mu, sigma^2) at each candidate point.
    """
    n_samples = 200
    
    # Ensure sigma is positive to avoid issues with sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal samples
    # Shape: (num_candidates, n_samples)
    z = np.random.normal(size=(mu.shape[0], n_samples))
    
    # Transform to the posterior distribution
    # y ~ N(mu, sigma^2)
    y = mu[:, np.newaxis] + sigma_safe[:, np.newaxis] * z
    
    # Calculate the improvement for each sample
    # Improvement = max(0, f_best - y)
    improvement = np.maximum(0.0, f_best - y)
    
    # The Knowledge Gradient score is the expected improvement
    kg_scores = np.mean(improvement, axis=1)
    
    return kg_scores
