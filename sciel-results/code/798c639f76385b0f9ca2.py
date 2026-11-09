import numpy as np
import time

# Track state across calls within a single seed run
_state = {
    'call_count': 0,
    'seed': None,
    'start_time': None,
    'last_f_best': None
}

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean of the GP at 2000 candidate points
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation of the GP at 2000 candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Thompson sampling scores; the evaluator picks the candidate with the highest score
    """
    # Initialize state on first call
    if _state['seed'] is None:
        _state['seed'] = np.random.randint(0, 2**31 - 1)
        _state['start_time'] = time.time()
        _state['last_f_best'] = f_best
    
    _state['call_count'] += 1
    step = _state['call_count']
    
    # Ensure sigma is strictly positive
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Thompson Sampling: sample a single realization from the posterior
    # For minimization, we want to select the point that minimizes f(x)
    # The posterior is N(mu, sigma^2) for each candidate
    # We sample from this and pick the candidate with the lowest sampled value
    # Since the evaluator picks the highest score, we return negative samples
    
    # Use a deterministic random seed based on the step to ensure reproducibility
    # but different across steps
    rng = np.random.RandomState(_state['seed'] + step)
    
    # Sample from the Gaussian posterior: sample = mu + sigma * z
    z = rng.standard_normal(size=mu.shape)
    samples = mu + sigma_safe * z
    
    # For minimization, we want low values, so we return negative samples
    # Higher score = lower sampled value = more likely to be the minimum
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    _state['last_f_best'] = f_best
    
    return scores
