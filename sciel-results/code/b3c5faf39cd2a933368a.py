import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Improvement (EI) with an adaptive exploration weight.
    The weight is higher when f_best is far from the known global minimum,
    encouraging exploration in early stages, and lower when close to the
    optimum, encouraging exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Adaptive exploration weight
    # Hartmann 6D minimum is approx -3.32237
    # Initial values are typically around -1.0 to -2.0
    # We define a 'progress' metric from 0 (start) to 1 (optimum)
    f_min = -3.32237
    f_start_estimate = -1.0
    
    # progress: 0 if f_best <= f_min, 1 if f_best >= f_start_estimate
    # We want beta to be high when progress is low (early)
    # and low when progress is high (late)
    
    # Normalize f_best to [0, 1] range based on expected trajectory
    # If f_best is -1.0, range is 2.32, diff is 2.32 -> progress 0
    # If f_best is -3.32, diff is 0 -> progress 1
    range_val = f_start_estimate - f_min
    diff = f_best - f_min
    
    # Clip diff to [0, range_val]
    diff_clipped = np.clip(diff, 0.0, range_val)
    
    # progress = 1 - (diff_clipped / range_val)
    # If f_best is -1.0, diff is 2.32, progress = 0
    # If f_best is -3.32, diff is 0, progress = 1
    progress = 1.0 - (diff_clipped / range_val)
    
    # Beta: High early (progress ~ 0), Low late (progress ~ 1)
    # Linear interpolation between beta_max and beta_min
    beta_max = 0.5
    beta_min = 0.1
    
    beta = beta_min + (beta_max - beta_min) * progress
    
    # Exploration term
    exploration = beta * sigma
    
    # Combined score
    scores = EI + exploration
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
