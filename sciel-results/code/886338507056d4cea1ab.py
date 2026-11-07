import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: float, the lowest observed function value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher scores indicate better candidates.
    """
    # UCB for minimization:
    # We want to find the point with the lowest potential function value.
    # The optimistic estimate of the minimum is mu - beta * sigma.
    # To maximize this score (since higher is better for selection), we take the negative:
    # score_raw = -(mu - beta * sigma) = -mu + beta * sigma
    
    beta = 3.0
    
    # Calculate the raw UCB score
    ucb_scores = -mu + beta * sigma
    
    # Normalize by cost to account for the cost-awareness
    # cost is in [1, 2], so dividing by it amplifies cheaper points
    # Ensure cost is not zero (it's >= 1)
    scores = ucb_scores / cost
    
    # Handle any numerical issues (NaNs or Infs)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
