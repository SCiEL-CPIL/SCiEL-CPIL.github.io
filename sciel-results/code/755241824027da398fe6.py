"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural Change: Hybrid Raw EI with Normalized Cost Penalty.
    
    This approach combines the Raw Expected Improvement (EI) from the high-performing 
    inspirations (which preserves the absolute scale of improvement potential) with 
    the Normalized Cost Penalty mechanism from the current program (which ensures 
    the cost penalty is scale-invariant relative to the current candidate set).
    
    Score = EI(x) - alpha * Cost_norm(x)
    
    Where Cost_norm(x) = (cost(x) - min(cost)) / (max(cost) - min(cost)).
    
    Key Insight:
    - Raw EI is preferred over Normalized EI because it correctly handles the 
      transition from high-improvement (early) to low-improvement (late) phases.
    - Normalized Cost is preferred over Raw Cost because it prevents the penalty 
      from becoming disproportionately large or small if the cost distribution 
      of the 2000 candidates changes shape during the optimization loop.
    - The penalty weight alpha is tuned to match the effective penalty of the 
      best-performing additive models (Inspiration 1).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
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
    
    # Check time early
    if time.time() > deadline - 5.0:
        # Fallback to simple EI if running out of time
        return ei
    
    # Normalize Cost to [0, 1] range based on current candidate set
    # This borrows the robustness of the current program's cost handling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros(n, dtype=np.float64)
    
    # Define the penalty weight alpha
    # In Inspiration 1, lambda=0.1 with raw cost penalty (cost-1).
    # Typical cost range in [0,1]^3 for Hartmann 3D is [1.0, 2.0].
    # If cost_range is ~1.0, cost_norm is ~cost-1.
    # So alpha=0.1 should be comparable to lambda=0.1 in Insp 1.
    alpha = 0.1
    
    # Compute the final score
    # Raw EI (from Insp) - Normalized Cost Penalty (from Current)
    score = ei - alpha * cost_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
