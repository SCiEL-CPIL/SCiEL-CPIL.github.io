import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing Expected Improvement (EI) or UCB, we draw a single
    sample from the posterior distribution at each candidate point and use
    that sample as the score. Since we are minimizing, a lower sample value
    indicates a higher probability of being the next best point.
    
    However, the task requires "higher meaning evaluate next".
    So we return the negative of the sampled value, or simply the negative
    of the sample if we want to maximize the score to find the minimum.
    
    Actually, standard Thompson Sampling for minimization:
    1. Draw f(x) ~ N(mu(x), sigma(x)^2) for each x.
    2. Select x with the lowest f(x).
    
    To fit the interface "higher score = better", we return -f_sample.
    
    This is a single-sample approximation of the expected improvement logic,
    but structurally different as it relies on the stochasticity of the
    selection process over the BO loop rather than averaging improvements
    within a single step.
    
    Parameters:
    mu: array of predicted means (shape N,)
    sigma: array of predicted standard deviations (shape N,)
    f_best: best observed function value so far (unused in TS, but kept for signature)
    
    Returns:
    scores: array of scores (shape N,), higher is better
    """
    N = len(mu)
    if N == 0:
        return np.array([])
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw a single sample from the posterior at each candidate
    # z ~ N(0, 1)
    z = np.random.randn(N)
    
    # f_sample = mu + sigma * z
    f_sample = mu + sigma * z
    
    # We want to minimize f. The acquisition score should be higher for better candidates.
    # So we return the negative of the sampled value.
    scores = -f_sample
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
