import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement and cost-awareness.
    The cost is 1 + mean(x_scaled), so cheaper points are near the origin in [0,1]^2.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # For minimization, we want to find points where f(x) < f_best
    # Let z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware adjustment:
    # Cost is 1 + mean(x_scaled), ranging from 1 to 2.
    # We want to prefer cheaper evaluations, but not too aggressively.
    # Use a multiplicative penalty: score = EI / (cost^alpha)
    # Or additive: score = EI - beta * cost
    # 
    # Let's try a combination: scale EI by a cost-dependent factor.
    # The budget is 30, and we start with 5 points. Each point costs ~1-2.
    # So we can afford roughly 15-30 evaluations.
    # 
    # A good approach: EI * (1 / cost^gamma) where gamma controls cost sensitivity.
    # Let's try gamma = 0.5 to 1.0.
    
    # Alternative: Use UCBD (Upper Confidence Bound for Budget) or similar.
    # Or simply: EI / cost
    
    # Let's try a few variants and pick one. Since we can't test here,
    # I'll go with EI / cost^0.5 as a reasonable compromise.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost penalty: divide by cost to favor cheaper points
    # But don't be too aggressive. Use sqrt(cost) in denominator.
    score = EI / np.sqrt(cost_safe)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
