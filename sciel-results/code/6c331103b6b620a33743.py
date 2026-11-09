import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and Expected Improvement.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is positive when mu < f_best, which is what we want for minimization
    
    # Adaptive UCB-EI blending with uncertainty-scaled beta
    # When uncertainty is high, rely more on UCB (exploration)
    # When uncertainty is low, rely more on EI (exploitation)
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    if sigma_mean > 1e-6:
        # Normalize sigma to get relative uncertainty
        rel_sigma = sigma_safe / sigma_mean
        
        # Adaptive beta: higher when uncertainty is high
        # beta ranges from ~1.0 (low uncertainty) to ~3.0 (high uncertainty)
        beta = 1.0 + 2.0 * np.clip(rel_sigma, 0, 2) / 2.0
        
        # UCB component for minimization: -mu + beta * sigma
        ucb_score = -mu + beta * sigma_safe
        
        # Adaptive weights: when overall uncertainty is high, use more UCB
        # When uncertainty is low, use more EI
        overall_uncertainty = sigma_mean
        if overall_uncertainty > 0.01:
            # High uncertainty phase: explore more
            ucb_weight = 0.7
            ei_weight = 0.3
        elif overall_uncertainty > 0.001:
            # Medium uncertainty: balance
            ucb_weight = 0.5
            ei_weight = 0.5
        else:
            # Low uncertainty: exploit
            ucb_weight = 0.3
            ei_weight = 0.7
    else:
        # Very low uncertainty: exploit
        beta = 1.0
        ucb_score = -mu + beta * sigma_safe
        ucb_weight = 0.3
        ei_weight = 0.7
    
    # Normalize both components to have similar scales
    ucb_std = np.std(ucb_score)
    ei_std = np.std(ei)
    
    ucb_norm = ucb_score / (ucb_std + 1e-8)
    ei_norm = ei / (ei_std + 1e-8)
    
    # Combine with adaptive weights
    combined = ucb_weight * ucb_norm + ei_weight * ei_norm
    
    # Cost-aware: divide by cost to penalize expensive points
    # But we want to maximize, so we divide by cost
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
