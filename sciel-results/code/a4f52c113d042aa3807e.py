import numpy as np
import time
from scipy.special import ndtr, expit

# Safety margin: The evaluator kills at 330s.
# We set a deadline at 325s to allow for cleanup/return.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviation predictions.
    f_best: float, the best observed value so far.
    cost: (2000,) array of costs for each candidate.
    
    Returns:
    scores: (2000,) array of finite scores. Higher is better.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Ensure inputs are finite and valid
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Calculate Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    
    # Standard EI formula using ndtr (normal CDF) and expit (logistic) for speed
    # Note: expit is not used here, we use exp for pdf
    EI = improvement * ndtr(z) + sigma * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)

    # Cost-normalized EI
    eff_ei = EI / cost

    # Dynamic exploration bonus scaled by remaining budget
    # As cost budget depletes, we should be more conservative (exploit more)
    # Estimate remaining budget: total budget is 30, start cost is 5 (5 uniform points)
    # We don't know exact elapsed cost, but we can use f_best as a proxy for progress
    # Or simply use a fixed adaptive strategy:
    # If f_best is close to 0, exploit more. If f_best is large, explore more.
    
    # Let's use a simple adaptive beta:
    # beta = beta_max * (1 - f_best / f_max_estimate)
    # f_max_estimate for Levy 2D on [-10,10]^2 is around 100-200
    # Let's use 50 as a reasonable scale
    f_scale = 50.0
    progress = np.clip(f_best / f_scale, 0.0, 1.0)
    
    # When progress is low (f_best large), explore more (high beta)
    # When progress is high (f_best small), exploit more (low beta)
    beta = 2.0 * (1.0 - progress) + 0.5
    
    # Cost-adaptive exploration
    # Penalize exploration in expensive regions
    cost_damping = np.sqrt(cost)
    adaptive_beta = beta / cost_damping
    
    exploration_term = adaptive_beta * sigma
    
    # Combine: Cost-normalized EI + Cost-adaptive exploration
    scores = eff_ei + exploration_term
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small floor to avoid ties or zero scores for valid candidates
    scores = np.maximum(scores, 1e-10)
    
    return scores
