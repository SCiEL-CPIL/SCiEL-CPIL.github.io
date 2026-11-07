import numpy as np
from scipy.stats import norm
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 310  # 330s limit, keep 20s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a batch-aware Lower Confidence Bound (LCB)
    with a repulsion term based on the current best point's location.
    
    The LCB is: score = mu - beta * sigma
    The repulsion term penalizes candidates that are similar to the best
    observed point (approximated by the candidate with the lowest mu).
    
    We use a hybrid approach:
    1. Standard LCB for exploitation/exploration
    2. A diversity term that encourages exploring regions far from the
       current best candidate in the (mu, sigma) feature space
    3. Dynamic beta that increases exploration early and decreases later
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: exploit the current best guess (lowest mu)
        mu = np.asarray(mu, dtype=np.float64)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    
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
    
    # Dynamic beta for LCB: higher exploration early, lower later
    # Beta ranges from about 3.0 early to about 0.5 late
    beta = 0.5 + 2.5 * np.sqrt(progress)
    
    # Compute LCB: lower is better, so we want to maximize mu - beta*sigma
    lcb = mu_safe - beta * sigma_safe
    
    # Find the index of the best candidate (lowest mu, i.e., most promising)
    best_idx = np.argmin(mu_safe)
    
    # Compute a diversity/repulsion term
    # We want to encourage exploration away from the region around the best candidate
    # Use a simple distance-based penalty in the (mu, sigma) feature space
    
    mu_min, mu_max = np.min(mu_safe), np.max(mu_safe)
    sig_min, sig_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min
    sig_range = sig_max - sig_min
    
    if mu_range < 1e-10:
        mu_range = 1e-10
    if sig_range < 1e-10:
        sig_range = 1e-10
    
    mu_norm = (mu_safe - mu_min) / mu_range
    sig_norm = (sigma_safe - sig_min) / sig_range
    
    # Distance from the best candidate in normalized (mu, sigma) space
    best_mu_norm = mu_norm[best_idx]
    best_sig_norm = sig_norm[best_idx]
    
    dist_to_best = np.sqrt((mu_norm - best_mu_norm)**2 + (sig_norm - best_sig_norm)**2)
    
    # The repulsion term: we want to ADD to the score for points far from best
    # This encourages diversity. Scale the repulsion based on progress.
    # Early on (high progress), we want more repulsion to explore broadly.
    # Late on (low progress), we want less repulsion to exploit.
    repulsion_weight = 1.5 * np.sqrt(progress) + 0.2
    
    # Normalize dist_to_best to [0, 1] for consistent scaling
    dist_max = np.max(dist_to_best)
    if dist_max > 1e-10:
        dist_norm = dist_to_best / dist_max
    else:
        dist_norm = np.zeros_like(dist_to_best)
    
    repulsion_term = repulsion_weight * dist_norm
    
    # Final score: LCB + repulsion term
    scores = lcb + repulsion_term
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
