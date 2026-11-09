"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: array of shape (n,) with GP mean predictions.
        sigma: array of shape (n,) with GP standard deviations.
        f_best: float, the best (lowest) observed value so far.
        
    Returns:
        array of shape (n,) with UCB scores (higher is better).
    """
    # Ensure sigma is positive to avoid log issues in beta calculation if needed, 
    # though here we just use it linearly.
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate the number of evaluations. 
    # Since we don't have direct access to 't' (step count), we can approximate 
    # or use a fixed beta. However, a common heuristic when t is unknown but 
    # the process is sequential is to use a constant beta or derive it from 
    # the spread of sigma if possible. 
    # In many BO implementations, beta = sqrt(2 * log(1 + t)).
    # Without t, a common default for 1D problems is around 1.0 to 5.0.
    # Let's use a moderate exploration constant. 
    # A better proxy: if we assume the evaluator runs 30 steps, and we are 
    # somewhere in that, a fixed beta is safer than a wrong dynamic one.
    # Let's try beta = 1.5, which is a common starting point for UCBi.
    # Alternatively, we can estimate t from the magnitude of f_best if we knew 
    # the scale, but that's risky.
    
    # Let's stick to a standard UCB formula: mu + beta * sigma
    # For minimization, we want to maximize this.
    beta = 1.5
    
    # Calculate UCB scores
    scores = mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
