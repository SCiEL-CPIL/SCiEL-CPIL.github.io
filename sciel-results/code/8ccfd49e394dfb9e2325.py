"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Structural change: Use a cost-aware Expected Improvement (EI) combined with
    a budget-aware exploration-exploitation schedule. Instead of hard thresholding
    (which can be brittle), we use a smooth cost-normalized EI score with an
    adaptive exploration bonus that decays as the budget is consumed. We also
    incorporate a simple regret lower-bound estimate to guide exploitation.
    
    Key ideas:
    1. Cost-normalized EI: EI / cost, which directly optimizes improvement per dollar.
    2. Adaptive exploration: A UCB-like term scaled by remaining budget estimate.
    3. Budget tracking: Use f_best and cost to estimate progress and remaining budget.
    4. Smooth scoring: No hard thresholds; all candidates get a meaningful score.
    """
    t_start = time.time()
    deadline = t_start + 280  # 300s limit, keep 20s safety margin

    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-normalized EI
    eipc = ei / cost_safe

    # Estimate budget progress
    # Total budget is 30. Initial 5 points cost ~7.5 (avg cost 1.5).
    # So remaining budget after initial points is ~22.5.
    # As f_best approaches 0, we're making progress.
    # Use a combination of f_best and cost to estimate progress.
    
    # f_best scale: powell 4d values range from 0 (min) to ~10+ (random points)
    # progress: 1 when f_best is large (early), 0 when f_best is small (late)
    progress = f_best / (abs(f_best) + 1.0)
    # Clamp to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)

    # Adaptive exploration coefficient
    # Early: more exploration; Late: more exploitation
    # beta controls the weight of the UCB term
    beta = 0.2 + 0.5 * progress

    # UCB term for minimization: -mu + beta * sigma
    # Scale by 1/cost to prefer cheap exploratory points
    ucb_term = (-mu + beta * sigma) / cost_safe

    # Combine EIPC and UCB with adaptive weights
    # Early: weight UCB more; Late: weight EIPC more
    w_eipc = 0.7 + 0.3 * (1.0 - progress)  # Higher EIPC weight when late
    w_ucb = 0.3 + 0.3 * progress           # Higher UCB weight when early

    # Normalize weights to sum to 1
    w_sum = w_eipc + w_ucb
    w_eipc /= w_sum
    w_ucb /= w_sum

    # Final score
    score = w_eipc * eipc + w_ucb * ucb_term

    # Add a small deterministic component to break ties
    # Use the index-based tie-breaking to ensure reproducibility
    # But since we don't have indices, use a small perturbation based on mu
    # This helps avoid plateaus in the score landscape
    tie_break = 0.01 * (-mu) / cost_safe
    score = score + tie_break

    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)

    # Deadline check
    if time.time() > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
