import numpy as np

def acquisition(mu, sigma, f_best):
    # Structural Change: Use Lower Confidence Bound (LCB) for minimization.
    # We maximize the score, which corresponds to minimizing the LCB.
    # LCB = mu - beta * sigma
    # Score = -LCB = -mu + beta * sigma
    
    # Beta controls the exploration/exploitation trade-off.
    # A higher beta favors exploration (high sigma).
    beta = 1.5
    
    score = -mu + beta * sigma
    
    # Ensure scores are finite as per task contract
    # If any non-finite values exist, replace them with a very low score
    # to ensure they are not selected.
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
