"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    
    Instead of computing a deterministic score like EI or UCB, we sample a 
    function value from the posterior distribution N(mu, sigma^2) for each 
    candidate. The candidate with the lowest sampled value is selected. 
    This naturally balances exploration (high sigma -> wide spread -> chance 
    of low sample) and exploitation (low mu -> likely low sample).
    
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
        Since Thompson Sampling samples the objective value (lower is better),
        we return -sampled_value so that higher scores correspond to lower 
        objective values.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback to simple UCB if we're running out of time
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        mu_safe = np.asarray(mu, dtype=np.float64)
        return -mu_safe + 2.0 * sigma_safe
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Thompson Sampling: Sample from N(mu, sigma^2)
    # Use a fixed random state per call? No, the evaluator calls this fresh each step.
    # We just need to draw a sample.
    # Note: We want to MINIMIZE, so we want the sample to be as LOW as possible.
    # The score returned should be HIGHER for better candidates.
    # So, score = -sampled_value.
    
    # Generate standard normal samples
    # To ensure reproducibility isn't broken by external factors, we use the default RNG.
    # However, for a single call, we just need one draw.
    z = np.random.standard_normal(mu.shape)
    
    # Sampled objective value
    sampled_values = mu + sigma * z
    
    # Ensure sampled values are finite
    sampled_values = np.where(np.isfinite(sampled_values), sampled_values, f_best)
    
    # Return negative of sampled values so that higher score = lower objective
    scores = -sampled_values
    
    # Final safety check for finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
