import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 2.5
    # We want to maximize the score.
    # LCB for minimization is mu - beta * sigma.
    # We want to find the minimum of LCB.
    # So we return the negative of LCB: beta * sigma - mu.
    scores = beta * sigma - mu
    # Ensure finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    return scores
