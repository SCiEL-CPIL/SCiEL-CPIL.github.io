import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound for minimization.
    # We want to minimize f(x).
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We return a score where higher is better.
    # So we return -LCB(x) = -mu(x) + kappa * sigma(x).
    
    # Exploration coefficient.
    # Tuning kappa: 
    # kappa=3.0 was used in the parent.
    # Let's try 2.0 for a more balanced exploration/exploitation trade-off.
    # In multimodal landscapes like Hartmann 6D, too much exploration can lead to 
    # wandering, while too little gets stuck in local minima.
    kappa = 2.0
    
    # Calculate scores
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
