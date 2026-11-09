"""Model-and-optimize cost-aware acquisition: fit parametric EI/UCB blend to maximize predicted improvement per cost."""
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

def acquisition(mu, sigma, f_best, cost):
    """
    Model-and-optimize cost-aware acquisition.
    
    Fits a parametric cost-aware acquisition (blend of EI and UCB with tunable
    exploration parameter kappa) by maximizing predicted improvement per unit cost
    over a grid of kappa values, using the observed mu/sigma/cost as the surrogate.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple score quickly
    if not _elapsed_ok():
        sigma_safe = np.maximum(sigma, 1e-6)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Robust sigma: ensure it's positive and not too small
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Precompute EI (Expected Improvement for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / cost_safe
    
    # Precompute UCB base: -mu + kappa * sigma for various kappa
    # We'll optimize over kappa to maximize the "predicted improvement per cost"
    # The acquisition is: score_kappa = (w_ei * EI/cost + w_ucb * (-mu + kappa*sigma) / cost)
    # But we want to find the best kappa that maximizes the top candidate's score.
    
    # Strategy: For each candidate kappa, compute the full score vector,
    # then select the kappa that gives the highest score at the argmax position.
    # This is a model-and-optimize approach where we fit the exploration parameter.
    
    # Use a grid of kappa values (exploration parameter for UCB)
    # Scale kappa based on the data: typical sigma range
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-8:
        kappa_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    else:
        # Adaptive grid centered around typical exploration
        kappa_grid = np.linspace(0.3, 4.0, 14)
    
    # Also optimize the blend weight w between EI/cost and UCB/cost
    # Use a small grid of weights
    w_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Precompute normalized components for speed
    # EI/cost is already computed as ei_cost
    # UCB/cost for each kappa: (-mu + kappa * sigma_safe) / cost_safe
    
    # Normalize ei_cost to [0,1] once
    ei_c_min = np.min(ei_cost)
    ei_c_max = np.max(ei_cost)
    ei_c_range = ei_c_max - ei_c_min
    if ei_c_range > 1e-12:
        ei_c_norm = (ei_cost - ei_c_min) / ei_c_range
    else:
        ei_c_norm = np.ones_like(ei_cost)
    
    # For each (kappa, w) pair, compute score and track the best
    # To be efficient, we compute the score vector for each combination
    # and evaluate a "quality" metric: the top-5 average score (robust to outliers)
    
    best_score = None
    best_kappa = 1.0
    best_w = 0.7
    
    # Compute UCB/cost for each kappa efficiently
    # ucb_cost_k = (-mu + kappa * sigma_safe) / cost_safe
    # = (-mu / cost_safe) + kappa * (sigma_safe / cost_safe)
    neg_mu_over_cost = -mu / cost_safe
    sigma_over_cost = sigma_safe / cost_safe
    
    for kappa in kappa_grid:
        ucb_cost = neg_mu_over_cost + kappa * sigma_over_cost
        
        # Normalize ucb_cost to [0,1]
        ucb_c_min = np.min(ucb_cost)
        ucb_c_max = np.max(ucb_cost)
        ucb_c_range = ucb_c_max - ucb_c_min
        if ucb_c_range > 1e-12:
            ucb_c_norm = (ucb_cost - ucb_c_min) / ucb_c_range
        else:
            ucb_c_norm = np.ones_like(ucb_cost)
        
        for w in w_grid:
            score = w * ei_c_norm + (1.0 - w) * ucb_c_norm
            
            # Quality metric: top-5 average (robust)
            # Use partition for efficiency
            if len(score) > 5:
                top5 = np.partition(score, -5)[-5:]
                quality = np.mean(top5)
            else:
                quality = np.mean(score)
            
            if best_score is None or quality > best_score:
                best_score = quality
                best_kappa = kappa
                best_w = w
    
    # Final score with the optimized parameters
    ucb_cost = neg_mu_over_cost + best_kappa * sigma_over_cost
    ucb_c_min = np.min(ucb_cost)
    ucb_c_max = np.max(ucb_cost)
    ucb_c_range = ucb_c_max - ucb_c_min
    if ucb_c_range > 1e-12:
        ucb_c_norm = (ucb_cost - ucb_c_min) / ucb_c_range
    else:
        ucb_c_norm = np.ones_like(ucb_cost)
    
    score = best_w * ei_c_norm + (1.0 - best_w) * ucb_c_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
