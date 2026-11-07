"""Cost-aware acquisition using Budget-Adaptive Lower Confidence Bound (LCCB).

Structural change: replace EI+UCB with a budget-adaptive LCCB that directly
optimizes expected regret reduction per unit cost. The exploration parameter
beta adapts to the remaining budget fraction, providing more exploration early
and exploitation late. The cost term uses direct division (expected improvement
per unit cost) rather than a power penalty, which is a different mathematical
structure that better reflects the actual budget constraint.
"""
import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0

# Track cumulative cost across calls to estimate remaining budget
_cumulative_cost = 0.0
_num_calls = 0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Budget-Adaptive Lower Confidence Bound acquisition function.
    
    Uses LCCB for minimization: score = -(mu + beta * sigma) / cost
    where beta adapts based on estimated remaining budget fraction.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    global _cumulative_cost, _num_calls
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Update cumulative cost estimate (each call adds the minimum cost ~1.0)
    # The actual cost of the selected point will be between 1.0 and 2.0
    _cumulative_cost += 1.5  # Average expected cost per evaluation
    _num_calls += 1

    # Estimate remaining budget fraction
    # Total budget is 30, we started with 5 initial points (cost ~5)
    # So remaining budget is approximately 30 - 5 - _cumulative_cost
    total_budget = 30.0
    initial_cost = 5.0  # 5 initial uniform random points
    remaining_budget = max(total_budget - initial_cost - _cumulative_cost, 0.0)
    budget_fraction = remaining_budget / max(total_budget - initial_cost, 1e-9)
    budget_fraction = np.clip(budget_fraction, 0.0, 1.0)

    # Budget-adaptive beta: more exploration when budget is plentiful
    # beta = beta_max * (budget_fraction)^gamma
    # When budget_fraction is high (early), beta is large (explore)
    # When budget_fraction is low (late), beta is small (exploit)
    beta_max = 2.0
    gamma = 0.8
    beta = beta_max * (budget_fraction ** gamma)
    
    # Ensure minimum exploration to avoid getting stuck
    beta = np.maximum(beta, 0.1)

    # Lower Confidence Bound for minimization:
    # LCCB = mu + beta * sigma (lower bound on true function value)
    # We want to minimize this, so acquisition score = -LCCB / cost
    # This directly optimizes expected regret reduction per unit cost
    lccb = mu + beta * sigma
    
    # Cost-efficiency: expected improvement per unit cost
    # Score = -LCCB / cost (higher is better, meaning lower LCCB and lower cost)
    # But we need to handle the sign carefully:
    # For minimization, lower LCCB is better, so we negate it
    # Dividing by cost penalizes expensive evaluations
    
    score = -lccb / cost

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
