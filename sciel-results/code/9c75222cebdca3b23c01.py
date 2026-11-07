import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Thompson Sampling: sample from the posterior N(mu, sigma^2) for each
    candidate and select the one with the lowest sampled value.
    """
    # Generate random samples from the posterior for each candidate
    # Each candidate x has posterior f(x) ~ N(mu[i], sigma[i]^2)
    # We sample once per candidate to get a single estimate
    samples = mu + sigma * np.random.standard_normal(len(mu))
    
    # For minimization, we want the lowest sampled value
    # Return negative of the sample so that higher score = lower function value
    scores = -samples
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
