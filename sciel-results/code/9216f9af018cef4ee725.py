"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural change: Combine deterministic additive cost penalty (from the best
    inspiration, score 1.0028) with a mild exploration bonus based on uncertainty.
    
    The core insight from the best-performing approach is that a simple additive
    penalty (EI - lambda * cost) creates a clearer trade-off than multiplicative
    or normalized approaches. This allows the search to escape local optima in
    cheap regions by exploring slightly more expensive areas with higher expected
    improvement.
    
    We refine this by adding a small exploration bonus proportional to sigma,
    which encourages visiting uncertain regions without the aggressive bias of
    UCB. The exploration term is scaled down relative to EI to keep exploitation
    dominant while still providing a gentle push toward unexplored areas.
    
    Score = EI - lambda * cost + beta * sigma
    
    where lambda = 0.1 (cost penalty) and beta = 0.05 (exploration bonus).
    Both constants are small enough to not dominate the EI signal but large
    enough to provide meaningful guidance.
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
    
    # Check time
    if time.time() > deadline - 5.0:
        # Fallback to simple EI if running out of time
        return ei
    
    # Cost penalty: deterministic additive penalty
    # The cost is 1 + mean(x_scaled), so higher cost = more expensive
    # We want to penalize expensive points. Use a small lambda to gently steer
    # toward cheaper regions without dominating the EI signal.
    lambda_cost = 0.1
    cost_penalty = lambda_cost * cost
    
    # Exploration bonus: small bonus for uncertain regions
    # This encourages visiting areas where the GP is uncertain, but much
    # less aggressively than UCB would.
    beta_exploration = 0.05
    exploration_bonus = beta_exploration * sigma_safe
    
    # Combined score: EI - cost_penalty + exploration_bonus
    # Higher score means more attractive to evaluate
    score = ei - cost_penalty + exploration_bonus
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
