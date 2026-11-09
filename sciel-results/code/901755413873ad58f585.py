"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware budget-scaled LCB for minimization.
    # cost: shape (2000,) per-candidate evaluation cost (1 + mean(x))
    # mu, sigma: shape (2000,) GP mean/std in objective units
    # f_best: scalar, lowest observed value

    # Total budget for the search loop (from task: cost < 30).
    # Estimate remaining budget from current total cost.
    # cost here is the per-candidate cost array, but we don't know total spent.
    # However, the loop stops when total cost >= 30. We can infer progress
    # from f_best and the fact that initial 5 points cost ~5*1.5 = 7.5 avg.
    # A robust proxy: use the median of the cost array as typical cost,
    # and estimate remaining budget from a fixed total of 30.
    # Actually, we don't have total_cost as input. We only have per-candidate cost.
    # We need a structural way to know how much budget is left.

    # Since we can't access total_cost directly, use a heuristic:
    # The search has done some number of evals. Each eval costs ~1.5 avg.
    # With budget 30, we can do ~20 evals. Early = more exploration.
    # We'll use a fixed schedule based on a typical total budget.

    # Total budget constant
    total_budget = 30.0

    # Estimate how much budget has been used. We don't have it, so use a
    # conservative estimate: assume we're at the start of a batch.
    # The loop evaluates 1 point per batch, so between calls the total cost
    # increases by ~1.5. We can't track this across calls without state.

    # Alternative: use the cost array itself. If the current candidates have
    # high cost, we're likely later in the search (cheaper points already taken).
    # This is weak. Instead, use a fixed exploration schedule based on
    # a reasonable estimate of progress.

    # Simplest robust approach: use a fixed beta that balances exploration
    # and exploitation, scaled by remaining budget estimate.
    # Since we can't know remaining budget precisely, use a conservative
    # estimate: assume ~50% budget used on average.

    # Better: use the cost array to estimate. The cost of the *selected* point
    # each round is ~1.5. After k rounds, total cost ~ 7.5 + 1.5*k.
    # With budget 30, k can go up to ~15. We're at round k+1 now.
    # We don't know k. Use a fixed beta that works well on average.

    # Let's use a budget-aware approach: the exploration term should be
    # proportional to sqrt(remaining_budget / total_budget).
    # Estimate remaining budget: we'll use a fixed fraction. Since we can't
    # track state, use a conservative 0.5 (half budget remaining on average).

    # Actually, let's use a different structural approach:
    # Use the cost array to create a cost-aware exploration bonus.
    # Points with lower cost get more exploration credit.

    # LCB for minimization: mu - beta * sigma
    # Cost-normalized: (mu - beta * sigma) / cost^gamma
    # Higher = better (we want to minimize mu, so lower mu is better)
    # But we return higher = evaluate next, so we want high scores for good points.
    # Good points have low mu. So score = (f_best - mu + beta * sigma) / cost^gamma
    # This is like expected improvement with exploration.

    # Let's use: score = (f_best - mu) / cost^gamma + beta * sigma / cost^gamma
    # = (f_best - mu + beta * sigma) / cost^gamma

    # gamma = 0.5 (measured improvement)
    # beta: budget-aware. Use a fixed value that balances well.

    gamma = 0.5
    cost_safe = np.maximum(cost, 1e-6)

    # Beta: exploration weight. For LCB, typical beta is 2-4.
    # With budget-aware scaling, use a moderate beta.
    # Since we can't track remaining budget precisely, use a fixed beta
    # that works well across the search.
    beta = 2.5

    # Compute LCB-style score
    exploration = beta * sigma
    exploitation = f_best - mu
    score = (exploitation + exploration) / (cost_safe ** gamma)

    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)

    return score
# EVOLVE-BLOCK-END
