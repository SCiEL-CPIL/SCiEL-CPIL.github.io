import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Uncertainty-weighted Expected Improvement with Thompson Sampling.
    
    Combines deterministic Expected Improvement (EI) with a Thompson Sampling (TS) term.
    EI provides strong exploitation of the posterior mean relative to f_best.
    The TS term provides stochastic exploration.
    A key structural change is the "uncertainty weighting" of the EI term:
    EI is scaled by a factor that increases with posterior uncertainty (sigma).
    This acts as a repulsive force from low-uncertainty (exploited) regions,
    encouraging the search to jump to distant, unexplored basins, which is critical
    for escaping the dense local minima of the Michalewicz function.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Component 1: Expected Improvement (Deterministic Exploitation) ---
    # Standardized improvement
    z_ei = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # --- Component 2: Uncertainty Weighting (Repulsive Force) ---
    # Points with higher uncertainty are likely in unexplored regions.
    # We weight EI by a factor that increases with sigma to encourage exploration.
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-9:
        sigma_median = 1e-9
    
    # Normalized uncertainty
    sigma_norm = sigma_safe / sigma_median
    
    # Weight: Points with higher uncertainty get a higher weight.
    # This acts as a repulsive force from the low-uncertainty (exploited) regions.
    # The weight is 1.0 + 0.5 * sigma_norm. This means points with average uncertainty
    # get a weight of 1.5, while points with very high uncertainty get a larger boost.
    weight = 1.0 + 0.5 * sigma_norm
    
    weighted_ei = weight * ei
    
    # --- Component 3: Thompson Sampling (Stochastic Robustness) ---
    # Add a TS component to avoid getting stuck in a single deterministic path.
    # This helps with "unlucky" deterministic choices.
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    ts_score = -y_samples
    
    # Normalize TS score to be comparable to EI?
    # EI is in objective units. TS is in objective units.
    # But EI is usually smaller than the full function value range.
    # Let's scale TS down to be a secondary term.
    ts_weight = 0.5
    ts_score_scaled = ts_weight * ts_score
    
    # Combine the two components
    scores = weighted_ei + ts_score_scaled
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
