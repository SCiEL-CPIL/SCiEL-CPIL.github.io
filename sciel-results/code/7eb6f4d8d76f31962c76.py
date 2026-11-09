"""Parametric cost-aware acquisition: optimize exploration parameter to maximize predicted improvement per cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard (shared across calls in the loop)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s wall-clock limit per evaluation; use safety margin

def _elapsed_ok():
    """Return True if we have at least ~5s margin before the hard limit."""
    return (time.time() < _DEADLINE - 5.0)

def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement with numerical stability."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _compute_ucb(mu, sigma, kappa):
    """Compute UCB score (for minimization)."""
    sigma_safe = np.maximum(sigma, 1e-8)
    return -mu + kappa * sigma_safe

def acquisition(mu, sigma, f_best, cost):
    """
    Parametric cost-aware acquisition using optimized exploration parameter.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple score quickly
    if not _elapsed_ok():
        ei = _compute_ei(mu, sigma, f_best)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute base EI and UCB components
    ei = _compute_ei(mu, sigma, f_best)
    
    # Parametric approach: optimize kappa (exploration parameter) to maximize
    # the expected improvement per unit cost across the candidate set.
    # We sample a range of kappa values and pick the one that gives the best
    # aggregate score.
    
    # Sample kappa values from a reasonable range
    # Early in optimization, more exploration is beneficial; later, exploitation
    # Use the spread of sigma to adapt: if uncertainty is high, explore more
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive kappa range based on the current uncertainty landscape
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # If uncertainty is high relative to the objective scale, explore more
    obj_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = sigma_mean / obj_scale
    
    # Kappa range: lower bound encourages exploitation, upper bound encourages exploration
    kappa_min = 0.1
    kappa_max = 2.0 + 2.0 * min(uncertainty_ratio, 1.0)
    
    # Sample kappa values
    n_kappa_samples = 20
    kappas = np.linspace(kappa_min, kappa_max, n_kappa_samples)
    
    # For each kappa, compute the cost-normalized UCB and find the one that
    # gives the best balance. We use a composite metric:
    # - The top-10% candidates by UCB/cost should have high EI potential
    # - The overall score distribution should be well-spread
    
    best_kappa = 1.0
    best_metric = -np.inf
    
    # Precompute EI/cost for reference
    ei_cost = ei / cost_safe
    
    for kappa in kappas:
        ucb = _compute_ucb(mu, sigma, kappa)
        ucb_cost = ucb / cost_safe
        
        # Composite metric: 
        # 1. Mean of top 10% by UCB/cost (rewards finding good candidates)
        top_idx = np.argsort(ucb_cost)[-max(1, len(mu)//10):]
        top_mean = np.mean(ucb_cost[top_idx])
        
        # 2. Correlation between UCB/cost and EI/cost (alignment of exploration and exploitation)
        if np.std(ucb_cost) > 1e-12 and np.std(ei_cost) > 1e-12:
            corr = np.corrcoef(ucb_cost, ei_cost)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        # 3. Variance of scores (encourages diversity in recommendations)
        score_var = np.var(ucb_cost)
        
        # Composite: reward high top candidates, good alignment, and diversity
        metric = top_mean + 0.5 * corr + 0.1 * np.sqrt(score_var + 1e-12)
        
        if metric > best_metric:
            best_metric = metric
            best_kappa = kappa
    
    # Compute final score using the optimized kappa
    ucb = _compute_ucb(mu, sigma, best_kappa)
    ucb_cost = ucb / cost_safe
    
    # Blend with EI/cost for stability
    # The blend weight adapts: when EI is concentrated (low variance), trust EI more
    ei_cost_var = np.var(ei_cost)
    ucb_cost_var = np.var(ucb_cost)
    
    # If EI is more concentrated, weight it higher
    total_var = ei_cost_var + ucb_cost_var + 1e-12
    ei_weight = ei_cost_var / total_var
    ucb_weight = 1.0 - ei_weight
    
    # Final score: weighted blend, normalized for stability
    score = ei_weight * ei_cost + ucb_weight * ucb_cost
    
    # Normalize to [0, 1] for consistent scale
    s_min = np.min(score)
    s_max = np.max(score)
    s_range = s_max - s_min
    if s_range > 1e-12:
        score = (score - s_min) / s_range
    else:
        score = np.ones_like(score)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
