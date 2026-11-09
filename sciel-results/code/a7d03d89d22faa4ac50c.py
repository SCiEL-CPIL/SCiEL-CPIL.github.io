import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Adaptive Expected Improvement (EI) with a dynamic threshold (xi).
    The threshold xi decreases as f_best approaches 0, shifting the strategy
    from broad exploration (large xi) to fine-grained exploitation (small xi).
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Estimate the "stage" of optimization using f_best.
    # The Weierstrass function on [-0.5, 0.5] typically has initial values around 0.5-1.0.
    # We assume a reference initial value of 0.5 for scaling.
    # As f_best drops, we want to reduce the exploration pressure (xi).
    ref_val = 0.5
    # Clamp f_best to avoid negative xi or huge xi
    f_best_clamped = np.clip(f_best, 0.0, ref_val)
    
    # Dynamic xi: Linear decay from 0.1 to 0.001
    # When f_best is high (0.5), xi is ~0.1.
    # When f_best is low (0.0), xi is ~0.001.
    xi = 0.001 + 0.099 * (f_best_clamped / ref_val)
    
    # Calculate Z score for EI with threshold xi
    # Z = (f_best - mu - xi) / sigma
    Z = (f_best - mu - xi) / sigma
    
    # Clip Z to avoid numerical issues in norm.cdf/pdf
    Z = np.clip(Z, -10, 10)
    
    # Expected Improvement formula:
    # EI = (f_best - mu - xi) * Phi(Z) + sigma * phi(Z)
    # Note: Standard EI is (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # With threshold xi, it becomes (f_best - mu - xi) * Phi(Z) + sigma * phi(Z)
    # This is equivalent to shifting the target by xi.
    
    cdf_Z = norm.cdf(Z)
    pdf_Z = norm.pdf(Z)
    
    score = (f_best - mu - xi) * cdf_Z + sigma * pdf_Z
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
