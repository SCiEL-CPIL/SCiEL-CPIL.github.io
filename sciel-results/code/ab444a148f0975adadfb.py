import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
# The evaluator kills the program after 330s.
# We track the start time of the first call to ensure we don't exceed the limit.
# Since `acquisition` might be called multiple times in a loop, we use a module-level
# timestamp for the first call.
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin: 10s before 330s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI)
    and Probability of Improvement (PI) with a dynamic uncertainty floor.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Check deadline
    if _time_left() < 0.5:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = np.min(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0

    # Replace non-finite values in arrays to prevent propagation
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score: (f_best - mu) / sigma
    # If mu is very low (better than f_best), z is large positive.
    # If mu is high (worse than f_best), z is large negative.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu < f_best, (f_best - mu) > 0.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1] range for blending
    # This is a structural improvement: it ensures that the relative contribution
    # of EI and PI is controlled by weights, not by the absolute scale of EI.
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Add a dynamic uncertainty floor to encourage exploration when the GP is uncertain
    # This term helps escape local optima by boosting regions with high sigma,
    # even if EI is low (e.g., if mu is far from f_best but sigma is huge).
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        uncertainty_bonus = 0.1 * (sigma_safe / max_sigma)
    else:
        uncertainty_bonus = np.zeros_like(sigma_safe)
    
    # Blend EI, PI, and uncertainty bonus
    # Weights: 0.6 for EI (exploitation), 0.3 for PI (confidence), 0.1 for uncertainty (exploration)
    # This balance is tuned for Hartmann 3D where exploitation is key but some exploration is needed.
    blended = 0.6 * ei_norm + 0.3 * pi_norm + 0.1 * uncertainty_bonus
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # This ensures we prefer cheaper evaluations that offer similar improvement potential.
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final safety check
    if not np.all(np.isfinite(scores)):
        scores = np.zeros_like(scores)
        
    return scores
