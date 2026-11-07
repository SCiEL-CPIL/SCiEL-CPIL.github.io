import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # sigma_safe to avoid division by zero if we were using Z-scores, 
    # but for LCB we just multiply. Still, keep sigma positive.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate global uncertainty metric
    # Use the median or mean of sigma to gauge the exploration state.
    # Median is more robust to outliers.
    global_uncertainty = np.median(sigma_safe)
    
    # Adapt beta based on global uncertainty.
    # If global uncertainty is high, explore more (higher beta).
    # If global uncertainty is low, exploit more (lower beta).
    # Base beta is 1.0 (equivalent to 1-sigma lower bound).
    # Scale factor: if uncertainty is 0.1, add 1.0 to beta?
    # Let's use a linear scaling capped.
    # Hartmann 6D sigma values: 
    # Early on, sigma might be large (e.g., 0.5 - 1.0).
    # Late on, sigma might be small (e.g., 0.01).
    # Let's map [0, 0.5] uncertainty to beta [1.0, 5.0].
    
    # Normalize uncertainty. Assume max relevant uncertainty is 0.5.
    # If global_uncertainty > 0.5, cap it.
    u_norm = np.clip(global_uncertainty / 0.5, 0, 1)
    
    beta = 1.0 + 4.0 * u_norm
    
    # LCB Score: We want to maximize (mu - beta * sigma)
    # Wait, for minimization, we want low f.
    # Lower Confidence Bound for minimization: LCB = mu - beta * sigma.
    # We want to pick the point with the LOWEST LCB.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we should return -LCB = beta * sigma - mu.
    
    score = beta * sigma_safe - mu
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
