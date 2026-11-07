import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # Structural change from Expected Improvement (EI) to UCB.
    # UCB uses a linear exploration term (beta * sigma) instead of the 
    # integrated tail probability of EI. This can lead to more aggressive 
    # exploration of high-uncertainty regions, which may help escape 
    # local minima or narrow valleys if EI is too conservative.
    
    # Exploration constant
    beta = 2.5
    
    # Optimistic value: mu - beta * sigma
    # We want to minimize, so lower is better.
    # Gain relative to f_best: f_best - (mu - beta * sigma)
    # = f_best - mu + beta * sigma
    
    gain = f_best - mu + beta * sigma
    
    # Only consider points where the optimistic value is better than current best
    gain = np.maximum(gain, 0.0)
    
    # Normalize by cost
    # Cost is >= 1.0
    scores = gain / cost
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
