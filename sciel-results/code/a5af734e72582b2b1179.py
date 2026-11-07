"""Rewritten program with Cost-Effective Confidence-Weighted Improvement."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Confidence-Weighted Improvement strategy.
    
    This function computes a score for each candidate point based on the GP mean (mu),
    standard deviation (sigma), the current best function value (f_best), and the
    evaluation cost. The score is designed to be maximized.
    
    Strategy:
    - Combines the potential improvement (f_best - mu) with an exploration term (beta * sigma).
    - Divides by the cost to favor cheaper evaluations.
    - This structural change from Expected Improvement (EI) to a linear combination of
      mean and uncertainty (similar to UCB but anchored to f_best) encourages broader
      exploration in multi-modal landscapes like Rastrigin, while maintaining cost efficiency.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Exploration parameter. A higher beta encourages more exploration.
    # 2.0 is a reasonable default for balancing exploitation and exploration
    # in early-to-mid stages of optimization on multi-modal functions.
    beta = 2.0
    
    # Ensure cost is positive to avoid division issues
    # The problem states cost is 1 + mean(x_scaled), so it should be >= 1.
    # We add a small epsilon for safety.
    safe_cost = np.maximum(cost, 1e-8)
    
    # Calculate the score
    # (f_best - mu) is the potential improvement if we assume the true value is mu.
    # beta * sigma adds a bonus for uncertainty.
    # Dividing by cost normalizes the benefit by the expense.
    score = ((f_best - mu) + beta * sigma) / safe_cost
    
    # Handle any numerical issues (NaN, Inf)
    # We want finite scores. If a score is +inf, it's a very good candidate.
    # If -inf, it's very bad.
    # np.nan_to_num replaces NaN with 0, +inf with a large number, -inf with a small number.
    # However, we want to preserve the ordering.
    # A better approach: clip the values before nan_to_num or just replace NaN with -inf (bad)
    # and keep infs if they arise from division by zero (which we prevented).
    
    # Since we prevented division by zero, infinities shouldn't appear unless
    # mu or sigma are infinite. The GP should produce finite values.
    # But just in case:
    score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return score
# EVOLVE-BLOCK-END
