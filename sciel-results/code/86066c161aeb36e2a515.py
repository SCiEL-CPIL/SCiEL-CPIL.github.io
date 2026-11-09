import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # Posterior mean mu, std sigma.
    # We pick x that maximizes the "upper" bound of the potential improvement?
    # No, for minimization, we want the "lower" bound to be as low as possible.
    # Lower bound = mu - beta * sigma.
    # So we maximize (beta * sigma - mu).
    
    beta = 2.0
    scores = beta * sigma_safe - mu
    
    # Alternative: EI
    # z = (f_best - mu) / sigma_safe
    # ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # scores = ei
    
    # Let's try a blend or just UCB.
    # Given the "structural change" hint, UCB is a distinct family from EI.
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    if time.time() - start_time > 320:
         pass
         
    return scores
