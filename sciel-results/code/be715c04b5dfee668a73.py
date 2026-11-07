import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB)
    # Score = - (mu - beta * sigma)
    # We want to maximize this score (minimize mu - beta*sigma)
    beta = 5.0
    lcb_score = -mu + beta * sigma
    
    # Ensure finite
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, -np.inf)
    
    return lcb_score
