import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    # Start time for deadline guard
    start_time = time.time()
    
    # UCB for minimization:
    # We want to maximize the score.
    # Low mu is good. High sigma is good.
    # score = -mu + beta * sigma
    
    # Beta controls the exploration-exploitation trade-off.
    # Based on inspiration, beta=2.5 showed significant improvement over EI.
    beta = 2.5
    
    # Ensure inputs are numpy arrays for fast computation
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Calculate UCB scores
    # We add f_best to keep scores in a similar magnitude range, though it doesn't affect argmax
    scores = f_best - mu + beta * sigma
    
    # Handle numerical issues: replace non-finite values with a very low score
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard check (though this calculation is vectorized and very fast)
    elapsed = time.time() - start_time
    if elapsed > 0.1: # 100ms is a generous threshold for this simple op
        # If somehow this is slow, we still return the computed scores as they are ready
        # In a more complex iterative acquisition, we would abort here.
        pass
        
    return scores
