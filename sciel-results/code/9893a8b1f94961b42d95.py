"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a direct, principled blend of Expected Improvement and UCB,
    normalized by evaluation cost with a sqrt dampening to allow exploration
    of higher-cost regions.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware UCB
        beta = 1.5
        improvement = f_best - mu + beta * sigma
        scores = improvement / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Use a fixed beta for UCB, tuned to be slightly more exploitative
    # as suggested by the "tuned lower beta" finding
    beta = 0.8
    ucb_improvement = f_best - mu + beta * sigma
    
    # Base score: Cost-aware EI
    # We use EI as the primary driver, normalized by sqrt(cost)
    base_scores = ei / np.sqrt(cost)
    
    # Pareto-frontier-aware scoring
    # We identify points that are not dominated in (ei, sigma, 1/cost)
    # Dominance: i dominates j if ei[i] > ei[j] AND sigma[i] > sigma[j] AND cost[i] < cost[j]
    # Note: We use strict inequality to avoid self-domination issues in logic, 
    # but for Pareto front, usually >= is used for "at least as good" and > for "strictly better".
    # Here we want to find the "efficient" frontier.
    
    # To save memory/time, we can check dominance in chunks or use vectorized ops.
    # 2000x2000 is small enough for full vectorization.
    
    # Dimensions: 
    # 1. Higher EI is better
    # 2. Higher Sigma is better (uncertainty)
    # 3. Lower Cost is better (so higher 1/cost is better)
    
    ei_col = ei[:, None]
    ei_row = ei[None, :]
    
    sig_col = sigma[:, None]
    sig_row = sigma[None, :]
    
    cost_col = cost[:, None]
    cost_row = cost[None, :]
    
    # i dominates j if:
    # ei[i] > ei[j] AND sigma[i] > sigma[j] AND cost[i] < cost[j]
    # Using broadcasting:
    # dominates[i, j] is True if i dominates j
    
    # We need to be careful with memory. 2000*2000*3 booleans is ~12MB. Acceptable.
    
    better_ei = ei_col > ei_row
    better_sig = sig_col > sig_row
    better_cost = cost_col < cost_row
    
    dominates = better_ei & better_sig & better_cost
    
    # is_dominated[i] is True if any j dominates i
    is_dominated = np.any(dominates, axis=1)
    is_pareto = ~is_dominated
    
    # Boost scores for Pareto-optimal points
    # Penalty for dominated points
    # This encourages selection of points that offer a good trade-off
    # without being strictly worse than others in all aspects.
    
    boost_factor = 1.0 + 0.5 * is_pareto - 0.2 * is_dominated
    
    scores = base_scores * boost_factor
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
