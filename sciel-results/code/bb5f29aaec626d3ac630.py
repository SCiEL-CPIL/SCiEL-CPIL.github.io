import numpy as np
import time

# Global state for deadline tracking
_acq_start_time = None
_ACQ_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit
_TOTAL_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with exponential cost suppression.
    
    This function samples from the posterior distribution of the improvement
    (f_best - f(x)) and scales the score by an exponential cost gate.
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    global _acq_start_time
    
    # Initialize start time on first call
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    elapsed = time.time() - _acq_start_time
    if elapsed > _TOTAL_LIMIT - _ACQ_DEADLINE_MARGIN:
        # If we are extremely close to timeout, return zeros to be safe
        # This ensures we don't crash the evaluator.
        return np.zeros(len(mu))
    
    # Prevent division by zero or invalid standard deviations
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling:
    # The improvement I(x) = f_best - f(x) is normally distributed:
    # I(x) ~ N(f_best - mu(x), sigma(x))
    # We sample one realization from this distribution for each candidate.
    # This stochastic score naturally balances exploration (high sigma leads to wide samples)
    # and exploitation (high mean improvement leads to higher samples).
    
    # Generate random samples for the improvement
    # Using standard normal randoms scaled and shifted
    # z ~ N(0, 1)
    # I_sample = (f_best - mu) + sigma * z
    z = np.random.standard_normal(len(mu))
    sampled_improvement = (f_best - mu) + sigma_safe * z
    
    # The cost gate suppresses high-cost candidates.
    # We use an exponential penalty: exp(-k * cost)
    # k=1.5 was effective in the previous iteration.
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    # Final score: Sampled Improvement * Cost Gate
    # Higher is better.
    scores = sampled_improvement * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
