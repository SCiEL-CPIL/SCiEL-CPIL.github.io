# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for Thompson Sampling
_start_time = time.time()
_DEADLINE_MARGIN = 20  # seconds to keep before the 330s limit
_ts_seed_state = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and Thompson Sampling (TS).
    
    This approach combines the deterministic exploitation of EI with the stochastic
    exploration of TS. The TS component draws a single sample from the posterior
    predictive distribution, which naturally balances exploration and exploitation
    without manual tuning of exploration-exploitation trade-off parameters.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    global _ts_seed_state
    
    # Deadline guard
    if time.time() - _start_time > 330 - _DEADLINE_MARGIN:
        # Fallback: return simple EI if time is running out
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        f_best = float(f_best)
        sigma_safe = np.maximum(sigma, 1e-10)
        improvement = f_best - mu
        z = np.clip(improvement / sigma_safe, -10, 10)
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(improvement > 0, ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Component 1: Expected Improvement (deterministic exploitation)
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(improvement > 0, ei, 0.0)
    
    # Component 2: Thompson Sampling (stochastic exploration)
    # Draw a sample from the posterior predictive distribution
    # For minimization, we want to sample f(x) ~ N(mu, sigma^2)
    # The candidate with the lowest sampled value is the best bet
    # Return negative of sample so that higher score = lower sampled value
    if _ts_seed_state is None:
        _ts_seed_state = np.random.RandomState()
    
    # Draw standard normal samples
    z_ts = _ts_seed_state.standard_normal(n)
    # Sample from posterior: mu + sigma * z
    ts_sample = mu + sigma_safe * z_ts
    # For minimization, lower is better, so we return negative
    ts_score = -ts_sample
    
    # Normalize both components to have similar scale
    # EI is typically in the range [0, f_best - min(mu)]
    # TS scores can be anywhere, so we normalize by their standard deviation
    
    ei_scale = np.std(ei) if np.std(ei) > 1e-10 else 1.0
    ts_scale = np.std(ts_score) if np.std(ts_score) > 1e-10 else 1.0
    
    # Normalize
    ei_norm = ei / ei_scale
    ts_norm = ts_score / ts_scale
    
    # Combine: 50% EI, 50% TS
    # This gives a balanced approach: EI provides a strong exploitation signal
    # while TS provides stochastic exploration that can escape local optima
    blend_weight_ei = 0.5
    blend_weight_ts = 0.5
    
    scores = blend_weight_ei * ei_norm + blend_weight_ts * ts_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
