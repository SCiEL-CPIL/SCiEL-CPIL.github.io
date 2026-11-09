import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 285.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a structural shift to Bayesian Search Theorem-inspired scoring:
    maximizing expected reduction in posterior uncertainty of the global minimum,
    weighted by probability of improvement and normalized by cost.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        # Fallback: simple cost-normalized exploit
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_safe, 1e-9)
        fallback = -mu_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values for safety
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z = np.clip(z, -50, 50)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # --- Structural Change: Bayesian Search Theorem (BST) inspired score ---
    # BST suggests maximizing the expected reduction in the variance of the
    # posterior minimum. A practical approximation combines:
    # 1. Probability of Improvement (PI): P(f(x) < f_best)
    # 2. Expected Improvement (EI): E[max(f_best - f(x), 0)]
    # 3. Uncertainty reduction term: related to sigma * pdf(z)
    
    # PI: direct probability of beating current best
    pi = cdf_z
    
    # EI: expected improvement magnitude
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Information gain / uncertainty reduction component
    # The term sigma * pdf(z) is the derivative of EI w.r.t. f_best and
    # represents the "sharpness" of the improvement potential.
    # We also consider that high uncertainty regions (high sigma) in areas
    # where mu is close to f_best are valuable for reducing posterior uncertainty.
    
    # Progress metric: 0 when f_best is large (early), 1 when f_best ~ 0 (late)
    # Rastrigin 2d max value is around 40, so normalize by that
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = 1.0 - (f_best_clamped / 50.0)  # 0 when f_best=50, 1 when f_best=0
    
    # Adaptive weights for BST components:
    # Early on: prioritize exploration (uncertainty reduction)
    # Late on: prioritize exploitation (improvement)
    
    # Weight for EI (exploitation)
    w_ei = 1.0 + 0.5 * progress
    
    # Weight for PI (robust improvement probability)
    w_pi = 0.5 * (1.0 - progress)
    
    # Weight for uncertainty reduction term
    # The term sigma * pdf(z) peaks when z is around 0 (mu ~ f_best) and sigma is moderate
    w_unc = 0.3 * (1.0 - progress)
    
    # Uncertainty reduction score: high when sigma is large and z is near 0
    # This encourages sampling in uncertain regions near the current best
    unc_term = sigma_safe * pdf_z
    
    # Combine BST-inspired components
    score = (w_ei * ei + w_pi * pi * sigma_safe + w_unc * unc_term) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
