import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization
    # We want to minimize f(x), so we maximize the "optimistic" lower bound?
    # No, for minimization, we use the Lower Confidence Bound of the function value.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We select x that minimizes LCB(x).
    # Equivalently, we maximize Score(x) = -LCB(x) = beta * sigma(x) - mu(x)
    #
    # Let's use beta = 1.0
    beta = 1.0
    scores = beta * sigma - mu
    
    # Ensure finite scores
    # If sigma is 0, scores are -mu.
    # This is fine.
    
    return scores
