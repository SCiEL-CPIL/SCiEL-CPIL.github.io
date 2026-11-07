# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a cost-adjusted LCB formulation.
    
    The exploration term is scaled by 1/cost so that high-cost regions are
    penalized more heavily in the uncertainty bonus, while the mean term
    remains invariant. This creates a non-uniform exploration landscape that
    prioritizes high-uncertainty, low-cost areas early in the budget.
    """
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost is 1 + mean(x scaled to [0,1]), so cost >= 1
    # Ensure cost is at least 1e-9 to avoid division issues
    cost = np.maximum(cost, 1e-9)
    
    # LCB formulation: we want to minimize, so lower confidence bound is
    # mu - beta * sigma
    # To make it cost-aware, we scale the exploration term by 1/cost
    # Higher cost -> smaller exploration bonus -> less likely to be selected
    
    # We want higher acquisition scores to mean "evaluate next"
    # So we negate the LCB (since we're minimizing)
    # LCB = mu - beta * sigma / cost
    # Acquisition = -LCB = -mu + beta * sigma / cost
    
    # Choose beta to balance exploration and exploitation
    # A typical value for LCB in BO is around 2-5
    # Since we're dividing by cost (which is ~1-2 for this problem),
    # we can use a moderate beta
    
    beta = 2.5
    
    # Cost-adjusted LCB acquisition
    # The mean term (-mu) is invariant to cost
    # The exploration term (beta * sigma / cost) is penalized by high cost
    acquisition_scores = -mu + beta * sigma / cost
    
    # Ensure all scores are finite
    acquisition_scores = np.where(np.isfinite(acquisition_scores), acquisition_scores, -np.inf)
    
    return acquisition_scores
# EVOLVE-BLOCK-END
