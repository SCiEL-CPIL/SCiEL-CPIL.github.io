import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability that this point improves upon f_best
    p_improve = norm.cdf(z)
    
    # Expected improvement (for minimization)
    ei = (f_best - mu) * p_improve + sigma_safe * norm.pdf(z)
    
    # Expected value of the improvement magnitude given improvement occurs
    # E[improvement | improvement] = (f_best - mu) + sigma * phi(z)/Phi(z) for z > 0
    # But we can approximate the "conditional expected improvement" more simply
    
    # Structural change: Use a product-based scoring instead of additive normalization
    # score = EI * (1 + alpha * p_improve)
    # This amplifies points that both have high EI AND high probability of improvement
    # Unlike additive combination, this creates a multiplicative synergy
    
    alpha = 1.5
    scores = ei * (1.0 + alpha * p_improve)
    
    # Add a small exploration bonus for high uncertainty regions where p_improve is moderate
    # This encourages exploration in uncertain areas without dominating
    exploration_bonus = 0.1 * sigma_safe * (1.0 - p_improve) * p_improve
    
    scores = scores + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
