import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing the Expected Improvement (which is deterministic and greedy),
    we sample a value from the posterior distribution for each candidate point.
    We then select the candidate with the lowest sampled value.
    
    Since the evaluator selects the candidate with the HIGHEST score,
    we return the negative of the sampled values.
    """
    # Generate samples from the Gaussian posterior: y ~ N(mu, sigma^2)
    # sigma is the standard deviation
    samples = mu + sigma * np.random.randn(mu.shape[0])
    
    # We want to minimize the objective function.
    # The evaluator picks the argmax of the returned scores.
    # Therefore, we assign higher scores to lower predicted values.
    scores = -samples
    
    return scores
