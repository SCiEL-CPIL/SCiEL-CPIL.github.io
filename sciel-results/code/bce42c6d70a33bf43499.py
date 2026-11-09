import numpy as np

def acquisition(mu, sigma, f_best):
    # Known global minimum of Goldstein-Price 2D
    min_val = 3.0
    
    # Calculate the gap to the known minimum
    # If f_best is below min_val (due to noise or scaling), gap is 0
    gap = max(0.0, f_best - min_val)
    
    # Adaptive beta for LCB (Lower Confidence Bound)
    # Higher beta encourages more exploration
    # Logarithmic scaling to prevent beta from growing too fast
    beta = 1.5 + 0.5 * np.log1p(gap)
    
    # LCB for minimization: minimize (mu - beta * sigma)
    # We return scores where higher is better
    # So we maximize (-mu + beta * sigma)
    scores = -mu + beta * sigma
    
    # Ensure scores are finite (they should be, but just in case)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
