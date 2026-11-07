import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize the function, so we seek low LCB values.
    # The acquisition score should be higher for better candidates.
    # Score = -LCB(x) / cost(x) = (kappa * sigma(x) - mu(x)) / cost(x)
    
    # Kappa controls the exploration-exploitation trade-off.
    # A value of 2.0 is a standard choice for balanced exploration.
    kappa = 2.0
    
    # Calculate score
    # Higher score indicates a more promising candidate
    score = (kappa * sigma - mu) / cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
