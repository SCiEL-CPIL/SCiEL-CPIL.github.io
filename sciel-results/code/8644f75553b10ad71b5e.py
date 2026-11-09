import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (LCB) for minimization.
    # The score to maximize is -mu(x) + beta * sigma(x).
    # This encourages exploring regions with high uncertainty (large sigma)
    # and exploiting regions with low predicted value (low mu).
    
    # A beta value of 2.5 is a robust default for multimodal functions
    # like Goldstein-Price, providing more exploration than standard EI.
    beta = 2.5
    
    scores = -mu + beta * sigma
    return scores
