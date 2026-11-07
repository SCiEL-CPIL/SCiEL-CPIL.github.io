"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a quantile-based exploration strategy combined with
    Expected Improvement (EI) for exploitation.
    
    The strategy selects points that are in the top 5% of uncertainty (high sigma rank)
    but only if their mean is below the current best (promise). This decouples exploration
    from the specific magnitude of the GP variance, focusing instead on relative uncertainty.
    
    Parameters:
    mu: np.ndarray, shape (2000,), mean of the GP posterior
    sigma: np.ndarray, shape (2000,), std dev of the GP posterior
    f_best: float, current best observed value (minimum)
    
    Returns:
    scores: np.ndarray, shape (2000,), higher scores indicate higher priority
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Check if we're still within time budget for the quantile-based component
    if time.time() < deadline:
        # Compute quantile-based uncertainty bonus
        # Sort sigma to get ranks
        sigma_sorted = np.sort(sigma_safe)
        
        # Get rank of each sigma value (1-based, higher rank = more uncertain)
        # Use searchsorted to find where each sigma falls in the sorted array
        ranks = np.searchsorted(sigma_sorted, sigma_safe, side='right')
        
        # Normalize ranks to [0, 1]
        n = len(sigma_safe)
        rank_norm = ranks / n
        
        # Top 5% of uncertainty: rank_norm > 0.95
        # Only apply bonus if the mean is below current best (promise)
        promise_mask = mu < f_best
        
        # Quantile-based bonus: only for high-uncertainty points with promise
        # Scale the bonus so it's meaningful relative to EI
        # Use a smooth function that peaks at the top 5%
        # bonus = 0 for rank_norm < 0.9, linearly increases to 1 at rank_norm = 1.0
        bonus_weight = np.clip((rank_norm - 0.9) / 0.1, 0.0, 1.0)
        
        # Apply promise gate: only bonus if mu < f_best
        bonus = bonus_weight * promise_mask
        
        # Scale bonus relative to typical EI magnitude
        # Use a fraction of sigma for the bonus magnitude
        bonus_scale = 0.3 * sigma_safe
        
        quantile_bonus = bonus * bonus_scale
        
        # Combine: EI + quantile bonus
        scores = ei + quantile_bonus
    else:
        # Fallback to pure EI if time is running out
        scores = ei
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
