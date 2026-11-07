"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural change: Scale-Invariant Additive Cost-Penalized Expected Improvement.
    
    This approach combines the core EI computation with a scale-invariant cost penalty.
    Instead of using a fixed lambda penalty (which changes relative importance as the
    optimization progresses), we normalize both EI and cost to [0, 1] ranges within
    the current candidate set. This makes the penalty weight alpha scale-invariant.
    
    Score = EI_norm - alpha * Cost_norm
    
    Key insight: The raw EI scale changes as the model improves (f_best decreases, 
    sigma shrinks). A fixed lambda penalty becomes relatively more or less important 
    over time. By normalizing both EI and cost, the penalty weight remains consistent
    throughout the optimization loop.
    
    We use alpha = 0.12, which provides a moderate penalty that gently steers toward 
    cheaper regions without dominating the EI signal. The minimum is uniformly 
    distributed in the inner 80% of the domain, so we should not aggressively 
    penalize cost.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple fallback
    if n == 0 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative. Clip small negative values due to floating point errors.
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time - if running out of time, return raw EI
    if time.time() > deadline - 5.0:
        return ei
    
    # Normalize EI to [0, 1] range
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros(n, dtype=np.float64)
    
    # Normalize Cost to [0, 1] range based on current candidate set
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros(n, dtype=np.float64)
    
    # Define the penalty weight alpha
    # A smaller alpha means we are less sensitive to cost.
    # A larger alpha means we strongly prefer cheap points.
    # 0.12 suggests we are willing to pay up to 12% of the max possible gain in extra cost.
    alpha = 0.12
    
    # Compute the final score
    score = ei_norm - alpha * cost_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
