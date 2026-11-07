import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB).
    
    LCB(x) = mu(x) - beta * sigma(x)
    
    We use a beta of 2.5 to balance exploration and exploitation.
    A higher beta than the standard 2.0 encourages more exploration,
    which is often necessary to escape local minima in multimodal
    functions like Hartmann 3D within a small budget.
    """
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate LCB
    # For minimization, we want to minimize mu - beta * sigma
    # So the score to maximize is -LCB = -mu + beta * sigma
    # Or simply return -LCB and let the evaluator pick the max?
    # The prompt says: "Return 2000 finite scores, higher meaning evaluate next"
    # So we return scores where higher is better.
    # LCB value is low when good (low mu, high sigma).
    # So we want to return -LCB.
    
    beta = 2.5
    lcb_values = mu_safe - beta * sigma_safe
    
    # Score = -LCB
    scores = -lcb_values
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
