"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def _pareto_frontier(mu, sigma, cost, f_best):
    """
    Fast Pareto-frontier score using vectorized dominance counting.
    Points on the non-dominated frontier get high scores; dominated points
    get scores inversely proportional to how many points dominate them.
    """
    n = len(mu)
    
    # Objectives to maximize
    imp = f_best - mu        # predicted improvement
    unc = sigma.copy()       # uncertainty
    neg_cost = -cost.copy()  # negative cost (maximize = minimize cost)
    
    # Count how many other points dominate each point
    # Point i dominates point j if imp[i]>=imp[j], unc[i]>=unc[j], neg_cost[i]>=neg_cost[j],
    # and at least one is strict.
    
    # For efficiency, compute dominance count via broadcasting in chunks
    # n=2000, so full 2000x2000 boolean matrix is 4M entries = ~32MB, manageable
    
    # Vectorized: for each pair (i,j), check if i dominates j
    # imp_i >= imp_j for all i,j
    imp_ge = imp[:, None] >= imp[None, :]       # (n, n)
    unc_ge = unc[:, None] >= unc[None, :]       # (n, n)
    nc_ge = neg_cost[:, None] >= neg_cost[None, :]  # (n, n)
    
    # At least one strict
    imp_gt = imp[:, None] > imp[None, :]
    unc_gt = unc[:, None] > unc[None, :]
    nc_gt = neg_cost[:, None] > neg_cost[None, :]
    any_strict = imp_gt | unc_gt | nc_gt
    
    # i dominates j
    dominates = imp_ge & unc_ge & nc_ge & any_strict
    
    # Count dominators for each point j: number of i that dominate j
    dom_count = dominates.sum(axis=0)  # shape (n,)
    
    # Score: points with 0 dominators (Pareto frontier) get 1.0
    # Points with more dominators get lower scores
    # Use exponential decay for smoothness
    pf_scores = np.exp(-0.5 * dom_count)
    
    # Ensure the frontier points get exactly 1.0
    pf_scores[dom_count == 0] = 1.0
    
    return pf_scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Combines Pareto-frontier selection with cost-aware UCB/EI blending.
    
    Parameters
    ----------
    mu : array (2000,) GP mean
    sigma : array (2000,) GP std
    f_best : float best observed value
    cost : array (2000,) evaluation cost per candidate
    
    Returns
    -------
    scores : array (2000,) higher = evaluate next
    """
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s wall-clock limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0)
    
    n = len(mu)
    
    # --- Phase 1: Compute base acquisition scores ---
    
    # GP-UCB for minimization
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-9:
        mu_range = 1.0
    
    sigma_median = np.median(sigma)
    sigma_rel = sigma_median / mu_range
    
    # Adaptive beta: start aggressive, decay as we get more information
    # Use cost-based progress: total cost / max_cost (30) as progress indicator
    # We don't have total cost directly, but cost array gives per-point costs
    # Use sigma_rel as proxy for exploration need
    
    beta_max = 4.0
    beta_min = 1.0
    beta_t = beta_min + (beta_max - beta_min) * np.clip(sigma_rel * 2.0, 0, 1)
    
    # Cost-aware: reduce exploration when average cost is high
    cost_mean = np.mean(cost)
    cost_factor = 1.0 / (1.0 + 0.15 * (cost_mean - 1.0))
    beta_t *= cost_factor
    
    beta_cost = 0.8
    ucb_term = -mu + beta_t * sigma
    cost_normalizer = cost ** beta_cost
    
    # EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # --- Phase 2: Pareto frontier computation (fast vectorized) ---
    
    pf_scores = _pareto_frontier(mu, sigma, cost, f_best)
    
    # --- Phase 3: Combine ---
    
    # Adaptive blending: early in search (high sigma_rel) favor exploration (UCB),
    # later favor exploitation (EI)
    ei_weight = np.clip(1.0 - 2.0 * sigma_rel, 0.2, 0.8)
    ucb_weight = 1.0 - ei_weight
    
    # Normalize UCB scores to [0, 1] range
    ucb_min = np.min(ucb_term)
    ucb_max = np.max(ucb_term)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-9:
        ucb_norm = (ucb_term - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    # Cost-aware blended score
    blended = ei_weight * ei_norm + ucb_weight * ucb_norm
    blended_cost = blended / cost_normalizer
    
    # Pareto boost: multiply by (1 + boost * pf_score)
    # Frontier points (pf_score=1) get the highest boost
    pareto_boost = 0.3
    final_scores = blended_cost * (1.0 + pareto_boost * pf_scores)
    
    # Ensure finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
