import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to avoid log of non-positive values
    ei = np.maximum(ei, 1e-15)
    
    # Log Expected Improvement
    scores = np.log(ei)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This is a gentle nudge: points with high sigma get a small boost
    # The bonus is proportional to sigma * phi(z), which is the "uncertainty" part of EI
    # but scaled down significantly so it doesn't dominate the exploitation signal
    exploration_bonus = 0.05 * sigma * norm.pdf(z)
    
    # Combine: log-EI provides the main exploitation signal,
    # exploration_bonus adds a mild preference for uncertain regions
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
