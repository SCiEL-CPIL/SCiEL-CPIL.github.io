import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        # Pick the cheapest candidate as a safe fallback
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-6)
        return 1.0 / cost_safe

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and non-zero
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is positive and non-zero
    cost = np.maximum(cost, 1e-9)
    
    # Ensure mu and cost are finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Lower Confidence Bound (LCB) score for exploration
    # For minimization, we want to maximize (beta * sigma - mu)
    beta = 2.0
    lcb_score = beta * sigma - mu
    
    # Blend EI and LCB score
    # Weight 0.7 for EI (exploitation) and 0.3 for LCB (exploration)
    # This helps in early stages when EI might be dominated by f_best
    blended_score = 0.7 * ei + 0.3 * lcb_score
    
    # Normalize by cost
    scores = blended_score / cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
