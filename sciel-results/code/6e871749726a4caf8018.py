# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Weighted Lower Confidence Bound (LCB).
    
    Approach:
    1. Compute the Lower Confidence Bound (LCB) for minimization: mu - beta * sigma.
    2. Calculate the potential improvement of this bound over the current best: 
       improvement = max(0, f_best - LCB).
    3. Normalize by cost to get the value per unit cost.
    4. Add a small exploration term (sigma / cost) to break ties and encourage 
       exploration in high-uncertainty regions even if the LCB doesn't show 
       immediate improvement.
    
    This differs from Expected Improvement (EI) by using a quantile-based 
    bound (LCB) rather than the expected value of the improvement, which 
    often leads to more aggressive and diverse exploration.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Beta parameter for exploration. 
    # A higher beta encourages more exploration. 
    # 2.0 is a reasonable default for moderate budgets.
    beta = 2.0
    
    # Lower Confidence Bound for minimization
    lcb = mu - beta * sigma_safe
    
    # Potential improvement of the LCB over f_best
    # If LCB < f_best, we have potential improvement.
    improvement = np.maximum(0, f_best - lcb)
    
    # Cost-aware score: Improvement per unit cost
    score = improvement / cost_safe
    
    # Add a small exploration bonus to handle cases where improvement is 0 
    # but uncertainty is high, ensuring we don't get stuck in local optima 
    # due to ties in the primary score.
    # Scale the exploration term to be comparable to the improvement term.
    # Since f_best and mu are in objective units, and cost is dimensionless,
    # we normalize the sigma term by a scale factor. 
    # We can estimate a scale from the current best or just use a small constant.
    # Using sigma / cost directly might be too large or small.
    # Let's use a small weight.
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
