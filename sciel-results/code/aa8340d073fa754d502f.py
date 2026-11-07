import numpy as np

def acquisition(mu, sigma, f_best):
    # Minimum variance acquisition: score candidates solely by their posterior standard deviation.
    # This forces the GP to resolve uncertainty in unexplored regions.
    # For minimization, higher sigma means more uncertainty, so we prioritize those points.
    
    # Ensure sigma is non-negative (it should be, as it's a standard deviation)
    sigma = np.maximum(sigma, 0)
    
    # Return sigma as the score; higher sigma -> higher score
    scores = sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
