import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Effective Acquisition Function.
    
    Maximizes -q_{0.1}(x) / cost(x), where q_{0.1} is the 10th percentile
    of the predicted objective distribution at x.
    """
    # 10th percentile z-score
    z_01 = -1.28155
    
    # Predicted 10th percentile in objective units
    # q_0.1 = mu + z * sigma
    q_low = mu + z_01 * sigma
    
    # We want to minimize q_low.
    # Acquisition score = -q_low / cost
    # Higher score = better candidate
    
    # Safe division
    safe_cost = np.maximum(cost, 1e-12)
    
    scores = -q_low / safe_cost
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
