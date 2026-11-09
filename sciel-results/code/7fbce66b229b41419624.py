"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a pure Lower Confidence Bound (LCB) strategy.
    This approach maximizes (mu - beta * sigma), which is equivalent to minimizing
    the LCB. The beta parameter adapts over time to balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Deadline guard: if we're getting close to the limit, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Simple fallback: just use -mu (exploitation only)
        return -np.asarray(mu, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Time-adaptive beta for LCB
    # Start with high exploration (beta=3.0) and decay to lower exploitation (beta=0.5)
    # Total evaluations is 30. Initial 5 are random. We have 25 BO steps.
    # Estimate current step based on elapsed time or just use a fixed decay schedule.
    # Since we don't know the exact step count, we can approximate based on elapsed time.
    # However, a simpler heuristic is to just use a moderate constant or a very slow decay.
    # Let's use a constant beta that is known to work well for multimodal functions in 2D.
    # Beta = 2.0 is a standard starting point. 
    # To be more robust, let's try a slightly higher beta initially if we assume early steps.
    # But without step count, let's stick to a robust constant.
    # Actually, we can estimate the step from elapsed time if we assume uniform evaluation cost,
    # but that's risky. Let's use a constant beta = 2.0 which is a good balance.
    # The directive says "constant beta" in the strategy description, but also mentions 
    # "time-adaptive logic" in the contrast. The specific directive says: 
    # "Use a deterministic greedy selection that directly maximizes the lower confidence bound (LCB) 
    # defined as `mu - beta * sigma` with a constant beta... bypassing all normalization... or time-adaptive logic"
    
    # So, use a CONSTANT beta.
    beta = 2.0
    
    # LCB score: we want to minimize f, so we maximize the "optimistic" prediction.
    # The LCB is mu - beta * sigma. We want the point with the lowest potential value.
    # So the acquisition score should be such that higher is better.
    # If we define score = -(mu - beta * sigma) = -mu + beta * sigma, this is the standard UCB for minimization.
    # Wait, the directive says "maximizes the lower confidence bound (LCB) defined as mu - beta * sigma".
    # Maximizing (mu - beta * sigma) means picking points where the mean is high but uncertainty is very high?
    # No, for minimization, the Lower Confidence Bound is the optimistic estimate of how low the function could go.
    # LCB = mu - beta * sigma.
    # We want to find the point with the *lowest* possible value, so we want to *minimize* the LCB?
    # No, usually in BO for minimization:
    # We want to find x that minimizes f(x).
    # We use an acquisition function that is high where f is likely low.
    # UCB (Upper Confidence Bound) for minimization is often defined as -mu + beta * sigma.
    # This is equivalent to maximizing -mu + beta * sigma.
    # Alternatively, one can define LCB = mu - beta * sigma and *minimize* it.
    # The directive says: "maximizes the lower confidence bound (LCB) defined as mu - beta * sigma".
    # This phrasing is slightly confusing. If we maximize (mu - beta * sigma), we are looking for high mean, high uncertainty.
    # That would be for *maximization* of the function.
    # For *minimization*, we want to *minimize* the LCB (mu - beta * sigma), which is equivalent to *maximizing* -LCB = -mu + beta * sigma.
    # Let's re-read carefully: "maximizes the lower confidence bound (LCB) defined as mu - beta * sigma".
    # If the task is to *minimize* Michalewicz, and we use LCB = mu - beta * sigma, we should *minimize* this value to find the next point.
    # But the return value must be "higher meaning evaluate next".
    # So if we want to minimize LCB, we should return -LCB.
    # -LCB = -(mu - beta * sigma) = -mu + beta * sigma.
    # This is exactly the standard UCB formula for minimization.
    
    # So, score = -mu + beta * sigma.
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
