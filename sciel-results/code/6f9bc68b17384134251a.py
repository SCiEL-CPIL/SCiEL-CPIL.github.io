"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    
    Score = (beta * sigma - mu) / cost
    
    where beta adapts based on f_best: high exploration early, exploitation late.
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)
    cost_safe = np.maximum(cost, 1e-10)

    # Adaptive beta: high exploration when f_best is large, exploitation when small.
    # f_best is the lowest observed value (min 0 for three-hump-camel).
    f_best_clamped = max(float(f_best), 0.0)
    
    # Schedule: beta = 1 + a * log1p(f_best)
    # At f_best=0: beta=1 (pure exploitation)
    # At f_best=1: beta=1+a*0.69
    # At f_best=5: beta=1+a*1.79
    # At f_best=10: beta=1+a*2.40
    # Use a=3.0 for strong early exploration
    beta = 1.0 + 3.0 * np.log1p(f_best_clamped)

    # Cost-aware LCB for minimization
    # Higher score = more desirable to evaluate
    scores = (beta * sigma_safe - mu) / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)

    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        simple_scores = (1.5 * sigma_safe - mu) / cost_safe
        return np.where(np.isfinite(simple_scores), simple_scores, -1e9)

    return scores
# EVOLVE-BLOCK-END
