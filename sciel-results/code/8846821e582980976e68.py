import numpy as np
def acquisition(mu, sigma, f_best):
    # Quantile-based acquisition: Maximize the predicted 5th percentile
    # This is equivalent to minimizing the lower bound of the uncertainty interval
    # z-score for 5th percentile is approximately -1.645
    # We want the value to be as low as possible, so we maximize (mu - 1.645 * sigma)
    # Note: If we are minimizing f, and f ~ N(mu, sigma^2),
    # the 5th percentile is mu - 1.645 * sigma.
    # Higher values of this quantity mean a "lower" predicted minimum is likely?
    # Wait.
    # If I want to find the minimum, I want the function value to be low.
    # The 5th percentile is a low value.
    # If I pick the point with the *highest* 5th percentile, am I picking the point with the highest lower bound?
    # No, usually for minimization, we want the *lowest* predicted value.
    # Let's think.
    # UCB for minimization: mu - kappa * sigma. We maximize this score.
    # This selects points with low mu or high sigma.
    # The 5th percentile IS mu - 1.645 * sigma.
    # So maximizing this score is exactly the same as using a UCB with kappa = 1.645.
    # The current program uses EI.
    # Is kappa=1.645 better than EI for this specific problem?
    # EI is often robust. UCB with fixed kappa can be too exploratory or too exploitative.
    # However, the directive explicitly asks for this structural change.
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 5th percentile quantile
    scores = mu - 1.645 * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
