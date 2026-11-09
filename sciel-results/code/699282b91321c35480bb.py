import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Thompson Sampling
    # Sample from the posterior for each candidate
    # f_sampled ~ N(mu, sigma^2)
    # We want to minimize f, so we want low f_sampled.
    # Score should be high for low f_sampled.
    # score = -f_sampled = -mu - sigma * z
    
    z = np.random.randn(len(mu))
    scores = -mu - sigma * z
    
    # Safety check for deadline (though this function is very fast)
    if time.time() - start_time > 10:
         # Fallback to EI if something went wrong or took too long
         z_ei = (f_best - mu) / (sigma + 1e-9)
         scores = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
         
    return scores
