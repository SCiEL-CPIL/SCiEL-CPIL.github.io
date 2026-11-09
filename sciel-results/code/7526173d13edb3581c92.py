import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Samples from the posterior distribution N(mu, sigma^2) for each candidate.
    Since we want to minimize the objective, we select the candidate with the
    lowest sampled value. To conform to the 'higher score is better' interface,
    we return the negative of the samples.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)
    
    # Sample from the posterior
    # Use standard normal random variables
    samples = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # Negate because higher score should indicate a better (lower) objective value
    scores = -samples
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
