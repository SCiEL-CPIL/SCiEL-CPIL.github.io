"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Constrained Thompson Sampling logic.
    
    Instead of Expected Improvement (EI), which can be dominated by high-variance
    regions that are unlikely to beat f_best significantly, we use the probability
    of improvement (PI) with a margin (xi). This encourages the algorithm to find
    regions with a high probability of being significantly better than the current best,
    normalized by the cost of evaluation.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # We want to find points where Y < f_best - xi with high probability.
    # Let xi be the minimum improvement we care about.
    # For a 8d problem with range roughly [-0.8, 1.0], a small xi is appropriate.
    # The seed program uses xi=0 implicitly in EI. 
    # Using a small xi > 0 helps avoid getting stuck in local optima by requiring
    # a "significant" drop, but not too large to ignore small steps.
    xi = 0.05
    
    # Ensure sigma is positive to avoid division by zero in z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # The probability that Y < f_best - xi is:
    # P(Y < f_best - xi) = P((Y - mu) / sigma < (f_best - xi - mu) / sigma)
    # = Phi((f_best - xi - mu) / sigma)
    
    z = (f_best - xi - mu) / sigma_safe
    
    # Probability of improvement with margin xi
    prob_improvement = norm.cdf(z)
    
    # Cost-aware score: Probability of significant improvement per unit cost
    # We divide by cost to penalize expensive evaluations.
    # cost is given as 1 + mean(x scaled to [0,1]).
    # Since x is in [-1, 1]^8, scaled to [0,1] is (x+1)/2.
    # mean(x_scaled) is in [0, 1]. So cost is in [1, 2].
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = prob_improvement / cost_safe
    
    # Handle any non-finite values (shouldn't happen with safe divisions, but just in case)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
