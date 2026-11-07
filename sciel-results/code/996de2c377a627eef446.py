import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT:
        # Return a simple fallback to avoid timeout
        # In a minimization context, we want to avoid high costs if stuck, 
        # but returning 0 is neutral. Let's return something that favors low cost slightly 
        # or just zeros to be safe and fast.
        return np.zeros_like(mu)
    
    # UCB (Upper Confidence Bound) for minimization
    # We want to minimize f(x). The GP predicts mu(x) and sigma(x).
    # We are interested in points that have low mu or high sigma (uncertainty).
    # A common UCB for minimization is: score = -mu(x) + beta * sigma(x)
    # Alternatively, we can view it as maximizing the potential for improvement.
    # Let's stick to the standard exploration/exploitation trade-off.
    # Exploitation: low mu is good -> -mu
    # Exploration: high sigma is good -> +beta * sigma
    
    # Beta controls the trade-off. 
    # In the seed program, EI was used. EI is roughly proportional to (f_best - mu) * P + sigma * phi.
    # UCB is simpler and often more robust in low-noise regimes where EI saturates.
    # Let's use a beta of 2.0 as suggested in the prompt's strategy description.
    beta = 2.0
    
    # Calculate UCB score
    # Higher score means we should evaluate this point.
    # We want low mu and high sigma.
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring
    # Cost is in [1, 2].
    # We divide by cost to penalize expensive evaluations.
    # The prompt suggests a linear cost penalty (cost^1).
    safe_cost = np.maximum(cost, 1e-9)
    score = ucb / safe_cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
