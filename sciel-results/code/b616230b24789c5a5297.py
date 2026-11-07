import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 325.0

# Global iteration counter
_ITER_COUNT = 0

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function.
    
    Structural change: Replaces static Expected Improvement (EI) with
    UCB using a time-decaying exploration parameter. This encourages
    broad exploration in early iterations to escape local minima in
    high-dimensional spaces, shifting to exploitation as the number
    of evaluations increases.
    
    Score = -mu(x) + sqrt(beta_t) * sigma(x)
    where beta_t = C / t
    """
    global _ITER_COUNT
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return -mu + 0.01 * sigma

    _ITER_COUNT += 1
    t = _ITER_COUNT
    
    # Exploration parameter decay
    # C=10.0 provides strong initial exploration.
    # At t=1, beta=10, sqrt(beta)=3.16
    # At t=25, beta=0.4, sqrt(beta)=0.63
    C = 10.0
    beta = C / t
    sqrt_beta = np.sqrt(beta)
    
    # Compute UCB scores for minimization
    # Higher score = more promising
    # Low mu (good objective) -> High -mu
    # High sigma (uncertain) -> High sqrt_beta * sigma
    scores = -mu + sqrt_beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
