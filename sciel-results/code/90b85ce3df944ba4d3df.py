import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Rastrigin 2D BO.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using standard normal CDF
    pi = norm.cdf(z)
    
    # Expected Improvement for reference
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    
    # Adaptive weight: as f_best gets closer to 0 (the minimum),
    # we want to be more exploitative. Use a smooth transition.
    # When f_best is large (early), explore more.
    # When f_best is small (late), exploit more.
    # Use a logistic-like weight based on f_best
    # f_best for Rastrigin 2D is in [0, ~40]
    # Weight for exploitation (PI) vs exploration (EI)
    # As f_best -> 0, weight -> 1 (pure PI)
    # As f_best -> large, weight -> 0.5 (mix)
    
    # Use a smooth step: w = 1 / (1 + exp(-k * (f_best - threshold)))
    # When f_best > threshold, w ~ 0.5 (balanced)
    # When f_best < threshold, w ~ 1 (exploitative)
    # Let's use threshold = 1.0, k = 2.0
    threshold = 1.0
    k = 2.0
    w = 1.0 / (1.0 + np.exp(-k * (f_best - threshold)))
    # Clamp w to [0.3, 1.0] to always have some exploration
    w = np.clip(w, 0.3, 1.0)
    
    # Blend PI and EI
    # Normalize EI to be comparable to PI (EI can be > 1, PI is in [0,1])
    # Scale EI by a factor to make it comparable
    # Typical EI values might be 0.1-2.0, PI is 0-1
    # Use a scaling factor
    ei_scaled = ei / (1.0 + np.mean(ei))  # Normalize by mean EI
    
    # Blend: score = w * PI + (1-w) * EI_scaled
    blended = w * pi + (1.0 - w) * ei_scaled
    
    # Cost-aware score: Maximize per unit cost
    # Use a softer penalty for cost to encourage exploration in low-cost regions
    # while still preferring high acquisition values
    # cost is guaranteed to be >= 1.0
    # Use a power law with exponent < 1 to reduce the penalty of higher costs
    # This allows the acquisition to overcome moderate cost differences
    cost_penalty = cost ** 0.5
    
    score = blended / cost_penalty
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
