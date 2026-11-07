import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D.
    
    Uses GP-UCB (Upper Confidence Bound) with a time-varying, decreasing exploration parameter.
    For minimization, we use the lower confidence bound:
    LCB = mu - beta * sigma
    
    The exploration parameter beta is estimated from the posterior statistics to balance
    exploration and exploitation. As the optimization progresses (f_best approaches the
    minimum), beta decreases to focus on exploitation.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Estimate progress using the relationship between f_best and the posterior
    # mu_scale represents the typical scale of the function
    mu_scale = np.std(mu) + 1e-8
    
    # Progress estimate: how much lower is f_best compared to the mean of mu
    # If f_best is close to min(mu), we're in exploitation phase
    # Use a smooth sigmoid-like transition
    progress = np.clip((np.mean(mu) - f_best) / (2 * mu_scale), 0, 1)
    
    # Time-varying beta: start with higher exploration, decrease over time
    # beta starts at a value that encourages exploration and decreases as we progress
    # Using a form similar to sqrt(log(t+1)) but adapted to our progress estimate
    # At progress=0 (early), beta should be larger; at progress=1 (late), smaller
    beta = 2.0 * (1.0 - 0.5 * progress)
    
    # Lower Confidence Bound for minimization
    # We want to minimize, so we prefer points with low mu and high sigma (uncertainty)
    # LCB = mu - beta * sigma
    # Higher LCB score means more likely to find a good minimum
    scores = mu - beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are reasonably bounded
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
