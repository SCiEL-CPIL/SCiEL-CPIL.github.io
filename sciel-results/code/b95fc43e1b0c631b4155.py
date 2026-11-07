import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-normalized Expected Improvement (EI).
    
    Score = EI(mu, sigma, f_best) / cost^gamma
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # If mu < f_best, Z > 0 (exploitation)
    # If mu > f_best, Z < 0 (exploration)
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Using standard normal CDF and PDF
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    EI = np.maximum(EI, 0.0)
    
    # Cost penalty
    # Previous best used gamma=3.0. Let's try gamma=2.0 for EI
    # EI scales with sigma, UCB scales with sigma.
    # EI is more "damped" than UCB for large sigma?
    # Actually for large sigma (Z ~ 0), EI ~ sigma * phi(0) ~ 0.4 * sigma.
    # UCB ~ kappa * sigma.
    # So EI is effectively kappa_eff ~ 0.4.
    # If we want similar exploration strength to kappa=1.5 UCB, we might need to boost EI or adjust gamma.
    # However, EI is more robust.
    
    gamma = 2.0
    cost_safe = np.maximum(cost, 1e-12)
    cost_penalty = cost_safe ** gamma
    
    score = EI / cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        pass
        
    return score
