import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using adaptive Expected Improvement.
    
    The exploration parameter xi adapts based on posterior uncertainty:
    - When sigma is large relative to improvement potential, increase exploration
    - When the search has found good values, shift toward exploitation
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive xi: scale exploration based on current uncertainty
    # Compute a measure of how much improvement is potentially available
    # If f_best is much better than the mean, we're in exploitation mode
    # If uncertainty is high, we need more exploration
    
    # Use the median sigma as a robust measure of current uncertainty
    median_sigma = np.median(sigma_safe)
    
    # The gap between f_best and the best predicted mean
    # If mu is already close to or better than f_best, reduce exploration
    best_predicted = np.min(mu)
    improvement_gap = f_best - best_predicted
    
    # Scale xi: 
    # - When improvement_gap is small (we're close to optimum), use small/negative xi for exploitation
    # - When uncertainty is high, use larger xi for exploration
    # - Base xi on the ratio of uncertainty to improvement potential
    
    if improvement_gap > 1e-6:
        # Normalize uncertainty relative to improvement potential
        uncertainty_ratio = median_sigma / max(improvement_gap, 1e-6)
        # Map uncertainty ratio to xi in [-0.1, 0.5]
        # High uncertainty -> larger xi (more exploration)
        xi = -0.1 + 0.6 * min(uncertainty_ratio, 1.0)
    else:
        # Very close to optimum, focus on exploitation
        xi = -0.05
    
    target = f_best - xi
    z = (target - mu) / sigma_safe
    # Calculate Expected Improvement
    scores = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the limited evaluation budget
    exploration_bonus = 0.05 * (sigma_safe / (np.max(sigma_safe) + 1e-9))
    scores = scores + exploration_bonus
    
    return scores
