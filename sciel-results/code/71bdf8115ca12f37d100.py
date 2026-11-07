import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # The uncertainty interval is [mu - beta*sigma, mu + beta*sigma].
    # We pick the point with the lowest lower bound.
    # To convert to a "higher is better" score for selection:
    # score = mu - beta * sigma
    
    beta = 2.0
    scores = mu - beta * sigma
    
    return scores
