# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation

# Fixed seed for reproducible Thompson Sampling noise
# This makes the acquisition function deterministic given the GP posterior
_RNG = np.random.RandomState(42)

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Deterministic Thompson Sampling.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far (unused in TS, but kept for interface)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: UCB-like score
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Ensure sigma is strictly positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Generate standard normal noise using the fixed RNG
    # Note: We draw a new sample from the fixed RNG state each time?
    # No, if we use a fixed seed, we should probably draw the *same* noise vector
    # every time to make it truly deterministic relative to the index?
    # Or should we advance the RNG?
    # If we advance the RNG, it's still stochastic over time, just reproducible.
    # The directive says "using a fixed random seed".
    # Usually, this implies `np.random.seed(42)` at the start, but that's global state.
    # Using a RandomState object is better.
    # If we call `self._RNG.randn`, it advances the state.
    # If we want it to be *identical* every call (pure function of mu, sigma), we should reset or use a fixed vector.
    # "constructs a deterministic surrogate... using a fixed random seed"
    # This phrasing suggests the surrogate itself is fixed.
    # Let's use a fixed noise vector generated once.
    
    # We need to handle the case where the length of mu might change? 
    # The contract says mu has shape (2000,). So it's fixed.
    
    # Generate a fixed noise vector of length 2000
    # We can generate it once at module level or inside if not cached.
    # Let's cache it.
    
    global _NOISE_VECTOR
    if not hasattr(acquisition, '_noise_vector'):
        # Initialize with fixed seed
        acquisition._noise_vector = np.random.RandomState(42).randn(2000)
    
    noise = acquisition._noise_vector
    
    # Compute sampled objective values
    y_samples = mu + sigma_safe * noise
    
    # Convert to acquisition scores (higher is better)
    # We want to pick the point with the lowest y_sample.
    scores = -y_samples
    
    # Numerical cleanup: ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
