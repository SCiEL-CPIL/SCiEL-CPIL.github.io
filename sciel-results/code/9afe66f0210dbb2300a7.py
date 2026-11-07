import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Computes the GP-UCB acquisition function with a decaying exploration parameter.
    
    Parameters:
    mu : np.ndarray
        Mean of the Gaussian Process posterior, shape (N,)
    sigma : np.ndarray
        Standard deviation of the Gaussian Process posterior, shape (N,)
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    np.ndarray
        Acquisition scores for each candidate, shape (N,). Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to a simple UCB if we are running out of time
        sigma_safe = np.maximum(sigma, 1e-8)
        return -mu + 2.0 * sigma_safe

    # Ensure sigma is positive to avoid division by zero or log(0)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Determine the current step index t
    # We don't have t explicitly, but we can infer it from the number of evaluations?
    # No, we don't have access to the history length directly.
    # However, the problem states: "5 uniform random points... until 30 evaluations".
    # So there are 25 optimization steps.
    # We cannot know 't' directly from the function arguments.
    # Wait, the function signature is fixed: acquisition(mu, sigma, f_best).
    # We cannot pass 't'.
    # We must use a fixed beta or infer t from f_best? No, f_best is not monotonic enough.
    # We must use a FIXED beta or a beta that doesn't depend on t.
    
    # Re-evaluating the "time-varying" directive.
    # "Strategy to try in this change: Use a Gaussian Process Upper Confidence Bound (GP-UCB) acquisition function with a time-varying, decreasing exploration parameter beta_t..."
    # If we cannot access 't', we cannot implement a time-varying beta directly in the function signature.
    # Unless... we use a global counter?
    # The prompt says "Keep the function names and signatures...". It doesn't forbid global state.
    # We can maintain a global counter `_step_count` that increments on each call.
    
    global _step_count
    if not hasattr(acquisition, '_step_count'):
        acquisition._step_count = 0
        
    # Increment step count
    acquisition._step_count += 1
    t = acquisition._step_count
    
    # Total optimization steps is 25.
    # If t > 25, we are in the final phase or re-evaluations?
    # The loop runs until 30 evaluations total. 5 initial + 25 optimization.
    # So acquisition is called 25 times.
    
    # Decaying Beta:
    # Start high to explore the multimodal landscape (25 RBFs).
    # End lower to refine the best region.
    # Beta_t = Beta_start - (Beta_start - Beta_end) * (t-1) / (Total_Steps - 1)
    
    Total_Steps = 25
    Beta_start = 4.0
    Beta_end = 1.5
    
    if t > Total_Steps:
        beta = Beta_end
    else:
        progress = (t - 1) / max(Total_Steps - 1, 1)
        beta = Beta_start - (Beta_start - Beta_end) * progress
    
    # UCB for minimization:
    # We want to find low values.
    # The GP predicts mean mu and std sigma.
    # A point is "good" if mu is low.
    # A point is "uncertain" if sigma is high.
    # UCB score (to maximize) = -mu + beta * sigma.
    
    score = -mu + beta * sigma_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
