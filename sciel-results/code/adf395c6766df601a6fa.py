import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: Array of mean predictions, shape (N,)
        sigma: Array of standard deviation predictions, shape (N,)
        f_best: The best (lowest) observed function value so far.
        
    Returns:
        Array of scores, shape (N,). Higher scores indicate more promising candidates.
    """
    # f_best is the minimum value seen so far.
    # We assume the global minimum is 0.
    # As f_best approaches 0, we want to exploit more (reduce exploration).
    # As f_best is large, we want to explore more (increase exploration).
    
    # Clamp f_best to a reasonable range to control beta.
    # Weierstrass function values near the minimum are small.
    # If f_best is very large, we are far away.
    f_best_clamped = np.clip(f_best, 0.0, 1.0)
    
    # Adaptive beta:
    # When f_best is 0 (at optimum), beta = 1.0 (exploit)
    # When f_best is 1 (far away), beta = 3.0 (explore)
    beta = 1.0 + 2.0 * f_best_clamped
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The UCB criterion suggests picking x that minimizes: mu(x) - beta * sigma(x)
    # To use a "maximize score" interface, we return:
    # score(x) = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    # If sigma or mu contains NaN/Inf, replace with -inf so they are not picked
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
