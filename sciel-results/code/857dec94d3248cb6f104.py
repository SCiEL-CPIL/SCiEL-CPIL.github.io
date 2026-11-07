import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    Maximizes the score: -mu + beta * sigma.
    This balances exploitation (low mu) and exploration (high sigma).
    """
    # Beta controls the exploration-exploitation trade-off.
    # A value of 2.0 is a common heuristic for 3D problems.
    beta = 2.0
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Calculate score
    # We want to minimize the objective, so we prefer low mu.
    # We want to explore, so we prefer high sigma.
    # Score = -mu + beta * sigma
    # Higher score means we prefer this candidate.
    
    score = -mu + beta * sigma
    
    # Ensure finiteness
    # The task requires finite scores.
    # Clip to a large finite number to satisfy "Return 2000 finite scores".
    # The relative ordering is preserved for typical ranges.
    max_val = 1e6
    score = np.nan_to_num(score, nan=0.0, posinf=max_val, neginf=-max_val)
    
    return score
