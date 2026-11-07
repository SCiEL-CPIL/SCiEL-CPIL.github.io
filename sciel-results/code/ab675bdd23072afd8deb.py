import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    This function computes the score for each candidate point based on the
    Lower Confidence Bound (LCB) of the GP posterior, divided by the evaluation
    cost. LCB encourages exploration of high-uncertainty regions (high sigma)
    and exploitation of low-mean regions (low mu), while penalizing expensive
    evaluations.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior in objective units.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    # Safety margins for numerical stability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Exploration-exploitation trade-off parameter
    # A higher beta encourages more exploration.
    # For multimodal functions like Griewank, a moderate beta is often effective.
    beta = 2.5
    
    # Lower Confidence Bound for minimization:
    # We want to minimize f(x). The "best" (lowest) expected value with confidence
    # is mu - beta * sigma.
    # To maximize the score (higher is better for selection), we invert this:
    # score proportional to (beta * sigma - mu)
    # This rewards high uncertainty (sigma) and low mean (mu).
    lcb_term = beta * sigma_safe - mu
    
    # Cost-aware: divide by cost
    score = lcb_term / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Optional: Clip extreme scores to prevent overflow issues in downstream sorting
    # Though np.sort handles -inf fine, it's good practice.
    # However, returning -inf for invalid points is safer than 0 if 0 is a valid score.
    # But the task says "Return 2000 finite scores".
    # So we must replace -inf with a finite number, e.g., a very low number.
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
