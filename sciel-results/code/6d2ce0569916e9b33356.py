import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Upper Confidence Bound
    # For minimization, we want to maximize the potential to find a lower value.
    # The Lower Confidence Bound (LCB) is mu - beta * sigma.
    # We want to pick points with low LCB.
    # So the acquisition score should be high when LCB is low.
    # Score = -LCB / cost = (beta * sigma - mu) / cost
    
    beta = 2.5
    lcb = mu - beta * sigma
    scores = -lcb / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
