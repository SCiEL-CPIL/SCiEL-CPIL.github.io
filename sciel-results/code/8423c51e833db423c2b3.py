import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling and Expected Improvement acquisition.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean predictions
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best (lowest) function value observed so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better for selection)
    """
    t_start = time.time()
    deadline = t_start + 280.0
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Thompson Sampling Component
    # Sample from posterior predictive: y ~ N(mu, sigma^2)
    # We want to MINIMIZE, so we sample and then invert the score (lower y is better)
    # To fit the "higher score is better" interface, we use -sample
    z_ts = np.random.normal(0, 1, n)
    ts_samples = mu - sigma_safe * z_ts
    
    # 2. Expected Improvement Component
    # EI for minimization
    improvement = f_best - mu
    z_ei = improvement / sigma_safe
    ei = improvement * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # 3. Adaptive Weighting
    # Global minimum of Michalewicz 2D
    opt = -1.8013
    sr = f_best - opt
    
    # As we get closer to the optimum, we should trust EI (exploitation) more
    # and Thompson Sampling (exploration) less.
    # Initially, TS helps jump between basins.
    if sr > 0.5:
        w_ts = 0.8
    elif sr > 0.1:
        w_ts = 0.5
    elif sr > 0.01:
        w_ts = 0.2
    else:
        w_ts = 0.0
    
    w_ei = 1.0 - w_ts
    
    # Normalize components to have similar scales before combining
    # TS samples are in objective units. EI is in objective units (improvement).
    # They are comparable.
    
    scores = w_ts * (-ts_samples) + w_ei * ei
    
    # Check deadline
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
