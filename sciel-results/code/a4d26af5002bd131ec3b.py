import numpy as np
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0 # 10s margin

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe default
        cost_safe = np.maximum(cost, 1e-9)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)

    # Safe sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Hybrid UCB + EI strategy
    # UCB component: -mu + beta * sigma (maximize this)
    # EI component: Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    beta = 2.0
    
    # UCB score (higher is better for minimization)
    ucb_score = -mu + beta * sigma_safe
    
    # EI calculation
    # Improvement = f_best - mu (if positive, we expect improvement)
    improvement = f_best - mu
    
    # Standardize
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate EI using normal CDF and PDF
    # For large negative z, EI -> 0
    # For large positive z, EI -> improvement
    from scipy.stats import norm
    
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Combine UCB and EI
    # Weight them: 0.5 * UCB + 0.5 * EI
    hybrid_score = 0.5 * ucb_score + 0.5 * ei
    
    # Cost-aware: divide by cost
    scores = hybrid_score / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
