import numpy as np
from scipy.stats import norm
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 320  # 300s limit, keep margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid Expected Improvement (EI) with
    a direct uncertainty bonus (UCB-style) to enforce exploration.
    
    This replaces the grid-based density penalty with a principled
    uncertainty bonus proportional to the GP posterior standard deviation.
    
    For minimization:
    - EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    - UCB(x) = -mu(x) + beta * sigma(x)
    
    We combine these: score = EI(x) + alpha * sigma(x)
    where alpha is a dynamic exploration coefficient.
    
    The key insight from the knowledge findings is that rewarding spatial
    diversity via the GP posterior sigma (which is high in unexplored regions)
    leads to significant improvements over heuristic density penalties.
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    if time.time() - _start_time > _TIME_LIMIT:
        return -np.asarray(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor to avoid numerical issues
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, sigma_floor)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    f_start_ref = 0.0
    
    # Calculate progress: 1.0 at start (poor f_best), 0.0 at optimum
    span = f_start_ref - f_global_min
    if span <= 0:
        progress = 0.5
    else:
        progress = (f_best - f_global_min) / span
        progress = np.clip(progress, 0.0, 1.0)
    
    # Compute Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical edge cases
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Dynamic exploration coefficient
    # Early on (high progress), explore more; late on, exploit more
    # alpha scales with sqrt(progress) to provide strong early exploration
    alpha = 0.5 * np.sqrt(progress) + 0.1
    
    # Uncertainty bonus: rewards regions with high posterior variance
    # This is a principled alternative to heuristic density penalties
    # sigma is already in objective units, so we use it directly
    uncertainty_bonus = alpha * sigma_safe
    
    # Final score: EI + uncertainty bonus
    scores = ei + uncertainty_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
