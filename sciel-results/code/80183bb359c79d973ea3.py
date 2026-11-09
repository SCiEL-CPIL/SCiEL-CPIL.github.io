"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function combining Expected Improvement (EI) with a cost-aware
    exponential penalty. The penalty lambda scales with optimization progress:
    as f_best decreases (progress), the penalty becomes weaker, allowing more
    exploitation of high-EI regions. Early on, high cost is penalized more heavily.
    
    mu, sigma: shape (2000,) GP mean and std in objective units
    f_best: lowest observed function value (scalar)
    cost: shape (2000,) evaluation cost for each candidate (in [1, 2] since x in [-2,2]^2 scaled to [0,1])
    
    Returns: shape (2000,) acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Safety margin: return within 300s
    deadline = start_time + 280.0
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (standard)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware penalty: exponential decay based on cost
    # cost is in [1, 2] (1 + mean of two coords scaled to [0,1])
    # Normalize cost to [0, 1] for the penalty: cost_norm = (cost - 1) / 1
    cost_norm = (cost - 1.0) / 1.0
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # Lambda scales with progress: when f_best is large (early), lambda is larger (more penalty)
    # When f_best approaches 0 (converged), lambda decreases, allowing more exploitation
    # f_best for rosenbrock is in [0, ...], so 1/(1+f_best) is in (0, 1]
    lambda_base = 2.0  # base penalty strength
    progress_factor = 1.0 / (1.0 + max(f_best, 0.0))
    # Early: f_best large -> progress_factor small -> lambda small (less penalty)
    # Late: f_best near 0 -> progress_factor ~1 -> lambda large (more penalty)
    # Actually we want: early on, be more exploratory (less cost penalty), late be more exploitative (more cost penalty)
    # So lambda should increase as we progress. Let's use:
    lambda_val = lambda_base * (1.0 - progress_factor) + 0.5 * progress_factor
    # Early (f_best large): lambda ~ 0.5 (mild penalty)
    # Late (f_best ~ 0): lambda ~ 2.0 (strong penalty)
    
    # Exponential penalty
    penalty = np.exp(-lambda_val * cost_norm)
    
    # Combined score: EI * penalty
    scores = ei * penalty
    
    # Add a small exploration bonus for high-uncertainty regions to avoid premature convergence
    # This helps when EI is small but sigma is large (unexplored areas)
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    exploration_bonus = 0.1 * (sigma_safe / (sigma_max + 1e-9)) * penalty
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen for this simple computation, but safety check
        pass
    
    return scores
# EVOLVE-BLOCK-END
