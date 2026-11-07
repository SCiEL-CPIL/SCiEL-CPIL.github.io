import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or invalid z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware Probability of Improvement (PI) with uncertainty confidence
    # PI = Phi((f_best - mu) / sigma)  -- bounded [0, 1], robust to outliers
    # Confidence term: sigma / (sigma + eps) to weight by uncertainty
    # Both normalized by cost to account for evaluation cost
    
    # Compute PI: probability that f(x) < f_best given GP posterior
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Uncertainty confidence: higher sigma means more potential for improvement
    # Use a smooth function that goes to 0 when sigma->0 and grows with sigma
    # sigma_conf = sigma / (sigma + 1.0) -- but this depends on scale
    # Better: use sigma directly as a weight, normalized
    # Since sigma is in objective units and can vary, use log(1+sigma) for robustness
    sigma_conf = np.log1p(sigma_safe)
    
    # Combine PI and uncertainty, both scaled by 1/cost
    # The PI term drives exploitation (where improvement is likely)
    # The sigma_conf term drives exploration (where uncertainty is high)
    # Weight: PI dominates when close to f_best, exploration when uncertain
    exploration_weight = 0.5
    scores = (pi + exploration_weight * sigma_conf) / cost
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
