import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    if _time_check():
        return 0.5 * sigma - mu
    
    n = len(mu)
    sigma = np.maximum(sigma, 1e-12)
    
    # Use pure UCB with a fixed beta that prioritizes exploration
    # Based on findings: pure UCB with beta=1.5 outperformed EI hybrids
    # in the 30-evaluation budget for this problem.
    # 
    # For Ackley 2D with length scale 0.2 on [0,1] (i.e., 0.2*10=2 in original units),
    # the function has many local minima. We need strong exploration early.
    #
    # Strategy: Use a beta that is high enough to explore broadly but not so high
    # that we ignore the objective entirely.
    #
    # The key insight from the improvement (0.9765 -> 1.093) was that pure UCB
    # with beta=1.5 prevented premature convergence to local minima.
    #
    # Let's use beta=1.5 as the base, with a slight adaptive component:
    # - Early on (f_best is large), use higher beta for more exploration
    # - Later (f_best is small), reduce beta slightly to focus exploitation
    #
    # But the finding suggests pure UCB works well, so keep it simple.
    
    beta = 1.5
    
    UCB = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(UCB), UCB, 0.0)
    
    return scores
