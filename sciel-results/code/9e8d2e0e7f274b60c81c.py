import numpy as np
import time

_T0 = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    This is a structural change from the deterministic Expected Improvement (EI)
    to a stochastic sampling approach. Thompson Sampling samples from the
    posterior distribution of the objective function and selects the point
    with the best sampled value. This is often more effective for multimodal
    functions as it naturally balances exploration and exploitation without
    requiring a manually tuned exploration constant.
    
    Args:
        mu: (2000,) posterior mean
        sigma: (2000,) posterior std
        f_best: float, lowest observed value (not directly used in TS, but kept for signature)
    
    Returns:
        (2000,) scores. Higher is better.
    """
    # Deadline check
    if time.time() - _T0 > _DEADLINE:
        # Fallback to pure exploitation (negative mean)
        return -mu

    # Ensure inputs are finite and sigma is positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    
    # Generate random samples from the posterior
    # z ~ N(0, 1)
    z = np.random.randn(len(mu))
    
    # Sampled values f ~ N(mu, sigma^2)
    f_sampled = mu + sigma * z
    
    # We want to minimize f, so we maximize -f
    scores = -f_sampled
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
