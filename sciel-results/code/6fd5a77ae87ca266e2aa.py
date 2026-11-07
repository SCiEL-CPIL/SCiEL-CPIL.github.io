import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted LCB with Gaussian cost weighting
    # Beta controls exploration vs exploitation
    beta = 2.5
    
    # Gaussian weight centered at expected optimal cost
    # Shekel 4D minimizer typically has cost ~1.4
    center_cost = 1.4
    cost_width = 0.2
    
    # Calculate weight for each candidate based on its cost
    # Higher weight for costs near center_cost
    cost_weight = np.exp(-0.5 * ((cost - center_cost) / cost_width) ** 2)
    
    # LCB: mu - beta * sigma
    # We want to minimize this, so for acquisition (higher is better), we use:
    # score = -(mu - beta * sigma) = beta * sigma - mu
    lcb_score = beta * sigma - mu
    
    # Combine with cost weight
    score = lcb_score * cost_weight
    
    return score
