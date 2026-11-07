"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid of Pareto-filtered UCB and Expected Improvement.
    """
    # Deadline guard: track elapsed time to avoid timeout
    t_start = time.time()
    deadline = t_start + 280  # 300s limit, keep 20s safety margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Return finite scores: use EI-like fallback
        sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    if sigma.shape != mu.shape:
        raise ValueError("mu and sigma must have the same shape")
    
    n = len(mu)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.clip(sigma, 1e-12, None)
    
    # --- Step 1: Pareto filtering ---
    # Mask out candidates whose predicted mean is worse than f_best by more than a tolerance.
    # Tolerance is adaptive: larger when f_best is far from 0 (early), smaller near optimum.
    # Weierstrass min is 0, so f_best <= 0 typically (or close to 0).
    # Actually f_best is the lowest observed value; for Weierstrass it can be negative or positive.
    # The minimum value is 0, so f_best >= 0? No, the task says "minimum value 0" meaning the global min is 0.
    # f_best is the lowest observed value, so f_best >= 0 in theory, but numerically could be slightly negative.
    
    # Adaptive tolerance: scale with the current uncertainty and f_best
    # Use a tolerance that allows exploration but filters out clearly bad regions
    # mu > f_best + tol means the mean is already worse than best, so unlikely to improve
    # Let's use a tolerance based on the typical sigma scale
    sigma_mean = np.mean(sigma_safe)
    tol = max(0.1 * sigma_mean, 1e-6)
    
    # Filter: keep candidates where mu is not too much worse than f_best
    # We want mu < f_best + tol (i.e., could potentially beat f_best)
    mask = mu < (f_best + tol)
    
    # Ensure at least some candidates remain; if all are filtered, relax
    if not np.any(mask):
        # Relax: keep the best 20% by mu
        k = max(1, int(0.2 * n))
        idx = np.argpartition(mu, k)[:k]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
    
    # --- Step 2: Compute scores for filtered candidates ---
    # Use a hybrid: blend of UCB and EI
    # UCB: mu - beta * sigma (minimization, so we want low mu and high sigma)
    # For minimization, UCB = mu - beta * sigma
    # EI for minimization: (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Adaptive beta: start high for exploration, decrease as f_best approaches 0
    # f_best is close to 0 when we're near the optimum
    # Use a decay based on f_best: if f_best is large, explore more
    # Weierstrass values are bounded; typical range might be [-5, 5] or so
    # Let's use beta that decays as f_best decreases
    # beta = beta_0 * (1 + f_best) / (1 + f_best_0) but f_best can be negative
    # Simpler: beta = beta_min + (beta_max - beta_min) * exp(-f_best / scale)
    # When f_best is large (far from optimum), beta is large (explore)
    # When f_best is small (near optimum), beta is small (exploit)
    beta_max = 3.0
    beta_min = 0.5
    # f_best should be >= 0 ideally, but clip for safety
    f_best_clip = max(f_best, 0.0)
    scale = 1.0  # Weierstrass scale
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best_clip / scale)
    
    # Compute UCB score (higher is better for selection)
    # For minimization, UCB = mu - beta * sigma gives lower values for promising points
    # But we need higher score = more likely to be selected
    # So UCB_score = -(mu - beta * sigma) = beta * sigma - mu
    ucb_score = beta * sigma_safe - mu
    
    # Compute EI score (higher is better)
    z = (f_best - mu) / sigma_safe
    ei_score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize both to [0, 1] for blending
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ei_min, ei_max = np.min(ei_score), np.max(ei_score)
    
    ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min + 1e-12)
    ei_norm = (ei_score - ei_min) / (ei_max - ei_min + 1e-12)
    
    # Blend: weight based on how close we are to optimum
    # Early: more UCB (exploration), Late: more EI (exploitation)
    # weight_ucb = 1 / (1 + f_best_clip) but normalize
    # Let's use: weight_ucb = 0.5 + 0.5 * exp(-f_best_clip / 0.5)
    weight_ucb = 0.5 + 0.5 * np.exp(-f_best_clip / 0.5)
    weight_ei = 1.0 - weight_ucb
    
    blended = weight_ucb * ucb_norm + weight_ei * ei_norm
    
    # Apply mask: set non-masked scores to very low
    scores = np.full(n, -1e10)
    scores[mask] = blended[mask]
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    if time.time() > deadline:
        # Return what we have; it's already computed
        pass
    
    return scores
# EVOLVE-BLOCK-END
