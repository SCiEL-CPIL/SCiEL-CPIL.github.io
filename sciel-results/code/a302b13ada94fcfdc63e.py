import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization of Weierstrass 1D.
    Uses a dynamic Expected Improvement (EI) with adaptive threshold.
    
    Args:
        mu: Array of shape (2000,) with mean predictions.
        sigma: Array of shape (2000,) with standard deviation predictions.
        f_best: The lowest observed value so far.
        
    Returns:
        Array of shape (2000,) with scores. Higher score means more promising.
    """
    # Handle edge cases
    if sigma is None or len(sigma) == 0:
        return -mu
    
    # Ensure sigma is non-negative and handle zeros
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Dynamic threshold: scale the improvement target by the current best value
    # When f_best is large (early), we want more exploration (higher threshold penalty)
    # When f_best is small (late), we want more exploitation (lower threshold)
    # The threshold xi controls how much improvement we expect over f_best
    
    # Adaptive xi: starts relatively high, decreases as f_best approaches 0
    # Use a logarithmic scaling to balance exploration/exploitation
    # |f_best| is used since f_best can be negative (Weierstrass min is 0, but values can be positive)
    # Actually, for minimization, f_best is the lowest observed value, which should be >= 0 for Weierstrass
    # But let's be safe and use abs
    
    f_abs = max(abs(f_best), 1e-6)
    
    # Dynamic xi: scales with the current best value
    # Early on (large f_best), xi is larger -> more conservative improvement expectation
    # Late on (small f_best), xi is smaller -> more aggressive
    xi = 0.01 * f_abs + 1e-4
    
    # Standard Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    
    # For minimization, improvement is when f(x) < f_best
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the CDF of standard normal and phi is the PDF
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Add a small exploration bonus scaled by sigma to encourage exploration in uncertain regions
    # This helps when EI is zero (no expected improvement) but uncertainty is high
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
