"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid LCB/EI acquisition function.
    
    Combines Expected Improvement (EI) for exploitation and Lower Confidence Bound (LCB) 
    for exploration. The scores are normalized before combination to ensure 
    balanced contributions. A bias towards LCB (exploration) is applied 
    to handle the multimodal nature of the Weierstrass function with a small budget.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute Lower Confidence Bound (LCB) score
    # For minimization, we want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # To convert to a "maximize" score, we use -LCB(x) = -mu(x) + beta * sigma(x)
    beta = 1.5  # Exploration parameter
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Normalize scores to zero mean and unit variance
    # This prevents one metric from dominating due to scale differences
    def normalize_scores(scores):
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-9:
            return scores - mean
        return (scores - mean) / std
        
    ei_norm = normalize_scores(ei)
    lcb_norm = normalize_scores(lcb_score)
    
    # 4. Combine with exploration bias
    # 0.7 weight on LCB (exploration) and 0.3 on EI (exploitation)
    # This structure was found effective in similar small-budget multimodal tasks
    scores = 0.7 * lcb_norm + 0.3 * ei_norm
    
    # Deadline guard
    if time.time() - start_time > 299:
        # Fallback to pure LCB if time runs out (unlikely)
        return lcb_score
        
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
