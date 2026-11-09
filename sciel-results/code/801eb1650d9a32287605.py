import numpy as np
def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration weight
    # Slower decay to maintain exploration in multimodal landscape
    exploration_weight = np.exp(-0.3 * f_best)
    
    # Quantile-normalized standard deviation
    # Use 25th percentile to scale sigma, highlighting regions with above-average uncertainty
    sigma_q = np.percentile(sigma, 25)
    if sigma_q < 1e-9:
        sigma_q = 1e-9
        
    sigma_term = sigma / sigma_q
    
    # Base score: negative mean (exploitation)
    base_score = -mu
    
    # Exploration bonus
    exploration_bonus = exploration_weight * sigma_term
    
    # Combined score
    score = base_score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
