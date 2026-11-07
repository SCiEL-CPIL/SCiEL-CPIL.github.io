"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Discounted Expected Improvement (CD-EI).
    
    Strategy:
    Instead of EI - lambda*cost (additive) or EI/cost (multiplicative), 
    we use a formulation that considers the *value* of the next evaluation 
    relative to the *total budget*.
    
    The cost function is 1 + mean(x). Since we don't have x, we only have 
    the cost array. The cost is roughly proportional to the "distance" from 
    the origin in the scaled space.
    
    A more principled approach for cost-aware BO is to maximize the 
    Expected Improvement per unit cost, but with a discount factor that 
    reflects the urgency of finding the minimum before the budget runs out.
    
    However, the current EI/cost is often too aggressive. 
    Let's try a "Thresholded" or "Regularized" version.
    
    Actually, let's look at the structure of Hartmann 3D. It has multiple local minima.
    The cost is 1 + mean(x). This means points near the origin (0,0,0) are cheap (cost ~1), 
    and points near (1,1,1) are expensive (cost ~2).
    
    If the global minimum is near (0,0,0), cheap points are good. 
    If it's near (1,1,1), expensive points are good.
    
    The previous strategy (EI - 0.1*cost) worked okay (1.0028).
    The seed strategy (EI/cost) was 1.0.
    
    Let's try a different structural change: **Knowledge Gradient (KG) approximation** 
    or a **Variance-Weighted EI**.
    
    Wait, the directive says "Make a STRUCTURAL change: a different design or modelling family...".
    
    Let's try **Lower Confidence Bound (LCB)** with cost penalty.
    LCB = mu - kappa * sigma.
    Score = LCB - lambda * cost.
    
    LCB is more exploratory than EI. EI is strictly about improvement over f_best. 
    LCB explores regions with high uncertainty even if f_best is low. 
    Given the budget constraint, exploration might be needed to find the global minimum 
    if we get stuck in a local one.
    
    Let's combine: 
    Score = (mu - kappa * sigma) - lambda * cost
    
    Or better: **Cost-Aware LCB**
    Score = (mu - kappa * sigma) / cost
    
    Let's try LCB / cost. This balances exploration (sigma) and exploitation (mu) 
    while normalizing by cost.
    
    Hyperparameters:
    kappa: Exploration constant. Usually 1.0-2.0 for standardized targets.
    Since targets are standardized in the GP, mu and sigma are in objective units.
    
    Let's try:
    scores = (mu - 1.5 * sigma) / cost
    """
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation if time is critical
        # Use a safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = -mu / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # LCB formulation
    # We want to minimize f(x).
    # LCB(x) = mu(x) - kappa * sigma(x)
    # Lower LCB is better.
    # So we want to maximize -LCB(x) / cost(x)
    # Score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    # Let's tune kappa. 
    # Standard EI uses the distribution. LCB is a heuristic approximation.
    # kappa = 1.0 is a common start.
    kappa = 1.0
    
    scores = (kappa * sigma_safe - mu) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
