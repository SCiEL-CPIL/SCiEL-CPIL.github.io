import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    For minimization, we want to find points where the Lower Confidence Bound (LCB)
    is low. LCB(x) = mu(x) - beta * sigma(x).
    To maximize a score where higher is better, we return:
    Score(x) = -LCB(x) = -mu(x) + beta * sigma(x)
    
    This structural change shifts from Expected Improvement (which can be overly
    greedy in high dimensions) to UCB (which explicitly balances exploitation
    and exploration via a linear combination of mean and uncertainty).
    """
    
    # Exploration parameter. 
    # A value of 1.0 is a standard heuristic for moderate exploration.
    # In 6D with few samples, we might need more exploration, so we can tune this.
    # Let's use 1.0 as a robust baseline for UCB.
    beta = 1.0
    
    # Calculate UCB score for minimization
    # Higher score means more promising (lower potential value or higher uncertainty)
    scores = -mu + beta * sigma
    
    # Handle potential numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
