import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Distance-to-Best weighted Expected Improvement acquisition.
    
    Combines Expected Improvement (exploitation) with a distance-based
    exploration bonus that encourages jumping to distant, unexplored regions.
    This is critical for escaping the dense local minima of the Michalewicz
    function within the time budget.
    
    The distance is computed in the standardized input space using the
    posterior variance (sigma) as a proxy for exploration status.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The best (lowest) objective value observed so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-10)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    Phi = 0.5 * (1.0 + _erf(z_clipped / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2.0 * np.pi)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.where(EI > 0, EI, 0.0)

    # Distance-based exploration bonus
    # Use sigma as a proxy for "distance from explored regions"
    # Points with high sigma are in unexplored areas
    # We want to boost points that are both promising (high EI) AND distant (high sigma)
    
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)

    # Combine EI and exploration
    # Weight the EI by a factor that increases with distance (sigma)
    # This creates a repulsive force from well-explored (low sigma) regions
    exploration_weight = 0.5 + sigma_norm  # Range [0.5, 1.5]
    
    score = EI * exploration_weight

    # Add a small pure exploration component for very uncertain regions
    # This ensures we occasionally jump to completely unexplored areas
    pure_exploration = 0.1 * sigma_norm * np.max(sigma_safe)
    score = score + pure_exploration

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score


def _erf(x):
    """
    Error function approximation using numpy.
    Uses the approximation from Abramowitz and Stegun.
    """
    # Use scipy if available, otherwise use a polynomial approximation
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        # Polynomial approximation for erf
        # Coefficients for the approximation
        sign = np.sign(x)
        ax = np.abs(x)
        # Use a rational approximation
        t = 1.0 / (1.0 + 0.3275911 * ax)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
        return sign * y
