import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses an adaptive Lower Confidence Bound (LCB) with a beta that scales
    with the ratio of current best value to typical uncertainty, providing
    strong exploration early and exploitation late.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Adaptive Beta:
    # Scale exploration with the potential for improvement relative to uncertainty.
    # Use median sigma for robustness.
    med_sigma = np.median(sigma_safe)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
    
    # Beta scales with f_best / med_sigma, floored at 0.5 and capped at 5.0
    # This provides:
    # - High exploration when f_best is large relative to uncertainty (early)
    # - Lower exploration when f_best is small (late, close to optimum)
    beta = max(f_best / med_sigma, 0.5)
    beta = min(beta, 5.0)
    
    # LCB score for minimization: maximize -mu + beta * sigma
    score = -mu_safe + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
