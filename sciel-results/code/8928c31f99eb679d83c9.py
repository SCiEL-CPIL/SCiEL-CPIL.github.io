import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses Thompson Sampling: sample a single realization from the GP posterior
    and return the negative sampled values as scores. This naturally balances
    exploration and exploitation without manual tuning of exploration parameters.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from N(mu, sigma^2)
    # We sample a single realization from the posterior for all candidates
    rng = np.random.default_rng()
    sample = mu + sigma * rng.standard_normal(len(mu))
    
    # Return negative sampled values as scores (higher score = lower predicted value)
    scores = -sample
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
