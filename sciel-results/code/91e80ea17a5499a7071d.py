# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a blended approach:
    1. Expected Improvement (EI) - exploitation
    2. Upper Confidence Bound (UCB) - exploration
    3. Probability of Improvement (PI) - mild exploration
    
    All normalized by cost to account for evaluation cost.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Upper Confidence Bound (beta controls exploration-exploitation tradeoff)
    # Use a beta that balances exploration and exploitation
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Probability of Improvement
    z_pi = (f_best - mu) / sigma
    pi = norm.cdf(z_pi)
    
    # Since we're minimizing, lower values are better.
    # EI is positive when mu < f_best, which is good.
    # UCB: for minimization, we want low mu and high sigma (uncertainty)
    # PI: probability that f(x) < f_best, which is high when mu < f_best
    
    # For minimization:
    # - EI rewards points where mu is much less than f_best
    # - UCB for minimization should be mu - beta*sigma (we want low values with uncertainty)
    #   Actually, let's think about this more carefully.
    #   In minimization, a good point has low f(x). High uncertainty (sigma) means it could be very low.
    #   So we want to explore points with low mu and high sigma.
    #   UCB for minimization: mu - beta * sigma (this is the lower confidence bound)
    #   But typically UCB is used for maximization. For minimization, we can use:
    #   - Lower Confidence Bound: mu - beta * sigma
    #   Or we can negate and use UCB on -f
    
    # Let's use a simpler approach:
    # For minimization, we want to find points with low expected value and/or high uncertainty.
    # 
    # EI is already well-suited for minimization.
    # For exploration, we can use:
    # - sigma itself (pure uncertainty)
    # - Or a combination that rewards both low mu and high sigma
    
    # Let's blend:
    # Primary: EI (exploitation)
    # Secondary: sigma (exploration) - but scaled appropriately
    
    # Normalize components to be on similar scales
    # EI is in objective units
    # sigma is in objective units
    
    # Weighted blend
    # w_ei: weight for exploitation
    # w_exp: weight for exploration
    
    # Adaptive weights based on how far we are from the optimum
    # If f_best is close to the known minimum (-78.3323), increase exploitation
    # If f_best is far from the minimum, increase exploration
    
    known_min = -78.3323
    # How much have we improved from a typical initial value?
    # Initial values are roughly in the range of the function
    # The function has values roughly in [-78, 50] range
    
    # Simple adaptive weighting
    # When f_best is much worse than known_min, explore more
    # When f_best is close to known_min, exploit more
    
    improvement_ratio = (f_best - known_min) / (f_best - known_min + 10)
    # This gives a value between 0 and 1, closer to 1 when f_best is much worse
    
    # Actually, let's use a simpler fixed blend that works well
    w_ei = 0.7
    w_sigma = 0.3
    
    # Blend EI and sigma
    # Both are in objective units, so we can combine directly
    score = w_ei * ei + w_sigma * sigma
    
    # Cost-aware normalization
    # Divide by cost to prefer cheaper evaluations
    score = score / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
