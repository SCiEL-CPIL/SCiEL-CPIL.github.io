import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard: 330s wall-clock limit, keep safety margin
    start_time = time.time()
    
    # EI for minimization: maximize E[f_best - f(x)]
    # f_best is the lowest observed value (best so far)
    # mu is predicted mean, sigma is predicted std
    # For minimization, EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are standard normal CDF and PDF
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to encourage diversity early on
    # The insight from other solutions suggests adding a sigma bonus helps
    # Use a moderate bonus that doesn't dominate
    exploration_bonus = 0.1 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
