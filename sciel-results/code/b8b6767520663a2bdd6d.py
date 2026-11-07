import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (numpy array): Posterior mean for each candidate.
    sigma (numpy array): Posterior standard deviation for each candidate.
    f_best (float): Best observed value so far.
    
    Returns:
    numpy array: Acquisition scores for each candidate. Higher is better.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle potential numerical issues (e.g., negative sigma due to numerical noise)
    sigma = np.maximum(sigma, 0.0)
    
    # UCB parameter. The prompt hints that beta=1 outperforms EI.
    beta = 1.0
    
    # For minimization, we want low mu and high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores. Replace any non-finite values with a very low score
    # so they are not selected, but the array remains finite.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # The problem states "Return 2000 finite scores". -inf is not finite.
    # We should replace -inf with a very small finite number.
    # However, if mu/sigma are valid, scores should be finite.
    # Let's use a large negative number as a fallback for non-finite inputs.
    min_float = np.finfo(np.float64).min
    scores = np.where(np.isfinite(scores), scores, min_float)
    
    return scores
