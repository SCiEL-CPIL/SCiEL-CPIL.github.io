"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track number of evaluations for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: if we're running long, just return something safe
    # (though with 2000 points and simple math, this shouldn't be an issue)
    
    # Adaptive UCB (Upper Confidence Bound) for minimization
    # We want to minimize, so we use Lower Confidence Bound (LCB) logic
    # Score = - (mu + beta * sigma)  =>  higher score means lower predicted value + uncertainty
    # Alternatively, we can use: -mu - beta * sigma
    
    # Adaptive beta: start with high exploration, decrease over time
    # Total evaluations is 30. First 5 are random.
    # We have 25 optimization steps.
    # beta = 1.0 / (1 + 0.1 * _eval_count)  # Example
    # Let's try a simpler decay.
    # Early on (low _eval_count), beta is high.
    # Later, beta is low.
    
    # A common choice for UCB in BO is beta = log(t) / 2 or similar.
    # Let's try: beta = 0.5 * (1.0 / (1.0 + 0.05 * _eval_count))
    # Or even simpler: beta decreases linearly from 1.0 to 0.1
    # But _eval_count starts at 1 for the first acquisition call (after 5 random points).
    # So _eval_count will go from 1 to 25.
    
    beta = 1.0 / (1.0 + 0.1 * _eval_count)
    
    # LCB score: we want to pick the point with the lowest expected value.
    # The acquisition function should return a score where higher is better.
    # So we negate the LCB.
    # LCB(x) = mu(x) - beta * sigma(x)
    # Score(x) = -LCB(x) = -mu(x) + beta * sigma(x)
    
    # Wait, the task says "minimizes sphere 1d".
    # f_best is the lowest observed value.
    # mu and sigma are in objective units.
    
    # Standard UCB for maximization: mu + beta * sigma
    # For minimization: mu - beta * sigma (LCB)
    # To maximize the acquisition score (higher is better to evaluate next):
    # We want the candidate with the lowest LCB.
    # So score = - (mu - beta * sigma) = -mu + beta * sigma
    
    # However, we need to be careful with the scale.
    # If mu is around 0 and sigma is small, this works.
    # If mu is large, -mu dominates.
    
    # Let's check the inspiration: "Adaptive UCB directly optimizes low predicted values with uncertainty bonus... Measured: 2.151 -> 68.12"
    # This suggests a significant improvement.
    
    # Let's use: score = -mu + beta * sigma
    # But wait, if mu is negative (which it can be, since sphere minimum is 0, but values can be positive),
    # and we are minimizing, we want low mu.
    # So -mu is high when mu is low.
    # beta * sigma encourages exploration.
    
    # Is there a risk of sigma being too large?
    # sigma is standard deviation.
    
    # Let's try this.
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
