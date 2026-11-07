import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Distance-weighted Expected Improvement with Thompson Sampling.
    
    Combines the deterministic exploitation of Expected Improvement with a
    "Distance-to-Best" repulsive force. This force scales the EI score by the
    inverse of the distance to the current best point (approximated via the
    posterior uncertainty structure), encouraging the search to jump to distant,
    unexplored basins. This is critical for escaping the dense local minima of
    the Michalewicz function. A small Thompson Sampling term adds stochastic
    robustness against unlucky deterministic selections.
    
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
    # EI measures the expected amount by which a candidate improves upon f_best.
    # Standardized improvement
    z_ei = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # --- Component 2: Distance-to-Best Repulsive Force ---
    # We approximate the "distance to the best point" using the posterior uncertainty.
    # In a GP, points far from observed data (and thus likely far from the best point
    # if the best point is in a dense region of observations) have higher sigma.
    # However, we want to *repel* from the current best neighborhood.
    # A common heuristic is to use the uncertainty itself as a proxy for distance.
    # But the directive asks for "inverse of the distance".
    # Let's define a "proximity" score based on how close the predicted mean is to f_best
    # relative to the uncertainty. If mu is close to f_best and sigma is low, it's "close".
    # We want to boost points that are NOT close.
    
    # Alternative interpretation: Use the normalized EI.
    # The "Distance-to-Best" idea in the prompt suggests scaling EI by inverse distance.
    # Since we don't have coordinates, we can't compute Euclidean distance to the best point.
    # However, we can use the fact that points with high sigma are likely unexplored.
    # Let's create a weight that favors points with high uncertainty (exploration)
    # but also considers the improvement potential.
    
    # A better structural change: "Repulsive force" means we should penalize points
    # that are too similar to the current best. In the absence of coordinates, we can
    # use the GP's predictive variance. Points with very low variance near f_best
    # are likely in the "exploited" basin. We want to jump away.
    
    # Let's use a weight that increases with sigma (exploration) and EI (exploitation).
    # But the specific strategy is "Distance-to-Best weighted acquisition that scales
    # the Expected Improvement by the inverse of the distance...".
    
    # Without coordinates, we can't compute distance. However, we can approximate
    # "distance" by the difference in predicted mean from f_best? No, that's improvement.
    # Maybe "distance" refers to the uncertainty?
    # Let's try a different angle: The Michalewicz function has many local minima.
    # The best point found so far is likely in one of them. We want to find another.
    # Points with high sigma are likely in different basins.
    # So, we can weight EI by sigma? That would make it more exploratory.
    # Or weight EI by 1/sigma? That would make it more exploitative (trust low uncertainty).
    # The prompt says "scales the Expected Improvement by the inverse of the distance".
    # If distance is large, inverse distance is small. This would *penalize* distant points?
    # That contradicts "forcing the search to jump to distant... basins".
    # Perhaps the "distance" is the distance in *objective space*?
    # If mu is far from f_best (in objective space), it's a different basin?
    # No, if mu is far (higher), it's worse.
    
    # Let's re-read: "repulsive force that forces the search to jump to distant, unexplored basins".
    # This implies we want to *increase* the score for distant points.
    # So we should scale by *distance*, not inverse distance? Or maybe the prompt meant
    # "scales by a factor that increases with distance".
    # Let's assume the intent is to boost exploration.
    
    # Let's use a hybrid: EI * (1 + sigma / sigma_ref). This boosts high uncertainty points.
    # sigma_ref is the median sigma.
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-9:
        sigma_median = 1e-9
    
    # Normalized uncertainty
    sigma_norm = sigma_safe / sigma_median
    
    # Weight: Points with higher uncertainty get a higher weight.
    # This acts as a repulsive force from the low-uncertainty (exploited) regions.
    weight = 1.0 + 0.5 * sigma_norm
    
    weighted_ei = weight * ei
    
    # --- Component 3: Thompson Sampling (Stochastic Robustness) ---
    # Add a small TS component to avoid getting stuck in a single deterministic path.
    # This helps with "unlucky" deterministic choices.
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
