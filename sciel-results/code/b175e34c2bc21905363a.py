import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of computing a deterministic expected improvement score, 
    this method samples a value from the posterior distribution for each candidate.
    The candidate with the lowest sampled value is selected by returning 
    the negative of the sample as the score (since higher scores are selected).
    
    This is a structural change from Expected Improvement (EI) as it uses 
    stochastic sampling rather than maximizing an expected value.
    """
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a sample from the posterior for each candidate
    # f(x) ~ N(mu(x), sigma(x)^2)
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # We want to minimize the function value.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    # The candidate with the lowest sampled value will have the highest (least negative) score.
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
