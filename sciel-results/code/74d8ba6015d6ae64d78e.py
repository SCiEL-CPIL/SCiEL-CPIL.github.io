import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Uncertainty-weighted Expected Improvement with Thompson Sampling.
    
    Combines deterministic Expected Improvement (exploitation) with a structural
    uncertainty-weighting term that acts as a repulsive force from exploited regions.
    A Thompson Sampling component provides stochastic robustness.
    
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
    
    # --- Component 2: Uncertainty-Weighted Exploration ---
    # The Michalewicz function has many local minima. Points with high posterior
    # uncertainty are likely in unexplored regions (different basins).
    # We weight the EI score by a factor that increases with uncertainty.
    # This acts as a repulsive force from low-uncertainty (exploited) regions.
    
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-9:
        sigma_median = 1e-9
    
    # Normalized uncertainty
    sigma_norm = sigma_safe / sigma_median
    
    # Weight: Points with higher uncertainty get a higher weight.
    # The 0.5 coefficient balances exploration vs exploitation.
    weight = 1.0 + 0.5 * sigma_norm
    
    weighted_ei = weight * ei
    
    # --- Component 3: Thompson Sampling (Stochastic Robustness) ---
    # Add a small TS component to avoid getting stuck in a single deterministic path.
    # This helps with "unlucky" deterministic choices and adds diversity.
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    ts_score = -y_samples
    
    # Normalize TS score to be comparable to EI?
    # EI is in objective units. TS is in objective units.
    # But EI is usually smaller than the full function value range.
    # Let's scale TS down to be a secondary term.
    ts_weight = 0.1
    ts_component = ts_weight * ts_score
    
    # --- Combine ---
    scores = weighted_ei + ts_component
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
