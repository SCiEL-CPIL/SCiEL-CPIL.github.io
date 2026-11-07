import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Add a small exploration bonus to f_best to encourage exploring
    # regions that are slightly worse than the current best but have potential.
    # This helps escape local minima in multimodal landscapes like Hartmann 6d.
    exploration_bonus = 0.1
    f_target = f_best + exploration_bonus
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_target - mu) / safe_sigma
    
    # Expected Improvement
    # EI = (f_target - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_target - mu) * cdf_z + safe_sigma * pdf_z
    
    # Cost-aware score: EI per unit cost
    # Ensure cost is not zero (though min cost is 1.0)
    safe_cost = np.maximum(cost, 1e-9)
    
    score = ei / safe_cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
