import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Ensure cost is positive to avoid division by zero or negative scores
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Weighted Upper Confidence Bound (UCB)
    # For minimization, we prefer low mean (mu) and high uncertainty (sigma).
    # The term (-mu + beta * sigma) rewards low predicted values and high uncertainty.
    # Dividing by cost penalizes expensive evaluations.
    
    # Beta controls exploration vs exploitation.
    # Higher beta -> more exploration.
    # For 8D multimodal functions, exploration is crucial.
    beta = 2.0
    
    scores = (-mu + beta * sigma) / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
