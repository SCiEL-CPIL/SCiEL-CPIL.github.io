"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Expected Improvement (EI) acquisition function with
    dynamic cost penalty and exploration balance.
    
    The score is EI(x) * exp(-lambda * (cost - 1)), where lambda is
    dynamically scaled by optimization progress. This creates a smooth
    cost barrier that favors cheaper evaluations when expensive ones
    don't offer proportional improvement.
    
    mu, sigma: shape (2000,) GP mean and std in objective units
    f_best: lowest observed function value (scalar)
    cost: shape (2000,) evaluation cost for each candidate
    
    Returns: shape (2000,) acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost penalty: lambda scales with progress
    # Early on (high f_best), we penalize cost more heavily
    # As we make progress (f_best decreases), we allow more expensive evaluations
    # lambda = 1 / (1 + max(f_best, 0))
    f_best_clipped = max(f_best, 0.0)
    lam = 1.0 / (1.0 + f_best_clipped)
    
    # Cost penalty: exp(-lambda * (cost - 1))
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    # This means the penalty ranges from exp(0)=1 (cost=1) to exp(-lambda) (cost=2)
    cost_safe = np.maximum(cost, 1.0)
    cost_penalty = np.exp(-lam * (cost_safe - 1.0))
    
    # Base score: EI * cost_penalty
    scores = ei * cost_penalty
    
    # Add a small exploration term to avoid completely ignoring high-uncertainty regions
    # This helps escape local minima in Rosenbrock's narrow valley
    # Exploration weight: small, scaled by sigma
    exploration_weight = 0.01
    exploration = exploration_weight * sigma_safe / cost_safe
    
    # Combine: primary exploitation + secondary exploration
    scores = scores + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    if time.time() - start_time > 280.0:
        return np.zeros_like(mu)
        
    return scores
# EVOLVE-BLOCK-END
