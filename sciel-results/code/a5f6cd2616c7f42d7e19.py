import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    """
    start_time = time.time()
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize f_best to [0, 1] range, assuming max sphere value is 25
    f_max = 25.0
    f_norm = np.clip(f_best, 0.0, f_max) / f_max
    
    # Adaptive beta parameter
    # High beta early (high f_best) for exploration, low beta late (low f_best) for exploitation.
    # Using sqrt to keep exploration high for a longer period.
    beta = 0.5 + 9.5 * np.sqrt(f_norm)
    
    # LCB score for minimization:
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We return scores such that higher is better.
    # So we return -LCB(x) = -mu(x) + beta * sigma(x)
    scores = -mu + beta * sigma_safe
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
         # Fallback to simple EI if something is wrong, though this is fast
         z = (f_best - mu) / sigma_safe
         from scipy.stats import norm
         cdf_z = norm.cdf(z)
         pdf_z = norm.pdf(z)
         ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
         return ei
         
    return scores
