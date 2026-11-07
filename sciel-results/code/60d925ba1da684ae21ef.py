"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s. We use a 325s deadline to be safe.
_start_time = time.time()
_DEADLINE = 325.0


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a simple score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (f_best - mu) / cost_safe

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Use Cost-Aware UCB (Upper Confidence Bound)
    # UCB is often more effective for multimodal functions like Rastrigin
    # with short budgets as it encourages broader exploration than EI.
    # Score = (f_best - mu + beta * sigma) / cost
    # beta controls the exploration-exploitation trade-off.
    # beta=2.0 is a strong exploration setting suitable for Rastrigin's local minima.
    beta = 2.0
    
    score = (f_best - mu + beta * sigma_safe) / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
