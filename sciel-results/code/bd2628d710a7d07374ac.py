import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_NORM_PDF_0 = 1.0 / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (shekel 4d).
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. Cost-aware weighting that favors lower-cost evaluations
    3. Uncertainty bonus for exploration in early stages
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (lower is better)
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Convert to numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-score for EI
    # For minimization, EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = _norm_pdf(z)
    
    # Expected Improvement for minimization
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware weighting
    # The cost is in [1, 2] range (1 + mean(x) where x in [0,1]^4)
    # We want to favor lower-cost evaluations, but not too aggressively
    # Use a soft penalty: weight = 1 / cost^alpha
    # alpha controls how strongly we penalize high-cost evaluations
    alpha = 0.5  # Moderate cost penalty
    
    cost_weight = 1.0 / np.power(cost, alpha)
    
    # Uncertainty bonus: encourage exploration when sigma is high
    # This helps in early stages when we have little information
    # Scale the bonus by the relative uncertainty
    uncertainty_bonus = 0.1 * sigma / np.maximum(np.abs(f_best), 1e-6)
    
    # Combine EI and uncertainty bonus
    base_score = EI + uncertainty_bonus
    
    # Apply cost weighting
    scores = base_score * cost_weight
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

def _norm_pdf(x):
    """Standard normal PDF, vectorized."""
    return _NORM_PDF_0 * np.exp(-0.5 * x * x)
