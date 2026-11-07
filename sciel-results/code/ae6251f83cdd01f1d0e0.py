"""Rewritten acquisition function for cost-aware BO on Levy 2D."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def acquisition(mu, sigma, f_best, cost):
    _ensure_start()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Cost is at least 1 by construction
    cost_safe = np.maximum(cost, 1.0)

    # Robust sigma floor: avoid division by zero and GP overconfidence
    mu_abs_scale = max(float(np.max(np.abs(mu))), 1e-3)
    sigma_floor = 1e-4 * mu_abs_scale
    sigma_safe = np.maximum(sigma, sigma_floor)

    # --- Expected Improvement (EI) ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Penalty for known-good / already-explored regions ---
    # If mu is close to f_best AND sigma is very small, we've already found
    # a good point here. Penalize to push search toward unexplored areas.
    # The penalty grows when both conditions hold strongly.
    # Normalize: how "good" is this point relative to f_best?
    # mu close to f_best means (f_best - mu) is small (or negative).
    # We want to penalize when mu <= f_best (already as good or better)
    # AND sigma is tiny (we're confident it's good).
    #
    # A clean way: compute a "certainty of being good" score.
    # If mu is well below f_best, it's a known good region.
    # If sigma is small, we're certain.
    # Penalty = weight * exp(-sigma_safe / (scale)) * max(0, f_best - mu)
    # But we want to penalize points that are ALREADY good, so the search
    # moves away. Actually, for minimization, we want to AVOID points where
    # mu is already low (close to or below f_best) and sigma is small.
    
    # Let's use: penalty proportional to how "exploited" a region is.
    # A region is exploited if sigma is small (low uncertainty) and mu is 
    # close to f_best (we've already seen good values here).
    
    # Scale for sigma: what's a "small" sigma?
    # Use a fraction of the mu scale.
    sigma_scale = max(0.05 * mu_abs_scale, 1e-3)
    
    # "Exploited-ness": high when sigma is small relative to sigma_scale
    # and mu is close to f_best.
    # Use a Gaussian-like decay in sigma and a linear term in how good mu is.
    
    # How good is mu? For minimization, lower is better.
    # improvement = f_best - mu (positive means mu is better than f_best,
    # but f_best is the best observed, so mu <= f_best always roughly)
    # Actually mu can be above f_best. We care about regions where mu is
    # close to f_best (i.e., we've already explored here and found good values).
    
    # Let's define: proximity = exp(-(f_best - mu)^2 / (2 * (0.1 * mu_abs_scale)^2))
    # This peaks when mu == f_best and decays as mu moves away.
    # But we only want to penalize when mu <= f_best (already good).
    # If mu > f_best, it's not a known-good region, so no penalty.
    
    prox_scale = max(0.1 * mu_abs_scale, 1e-3)
    diff = f_best - mu  # positive when mu < f_best (better than best)
    # Only penalize where diff > 0 (mu is better than or equal to f_best)
    diff_clipped = np.maximum(diff, 0.0)
    proximity = np.exp(-diff_clipped**2 / (2 * prox_scale**2))
    
    # Uncertainty factor: high when sigma is small
    uncert_factor = np.exp(-sigma_safe / sigma_scale)
    
    # Combined exploited-ness
    exploited = proximity * uncert_factor
    
    # Penalty weight: scale by the typical EI magnitude so it's comparable
    ei_scale = max(float(np.max(ei)), 1e-6)
    penalty_weight = 0.5 * ei_scale
    
    penalty = penalty_weight * exploited
    
    # --- Exploration bonus for high-uncertainty regions ---
    explore_scale = max(abs(f_best), mu_abs_scale, 1e-6)
    explore = sigma_safe / explore_scale
    
    # Combine: (EI - penalty + exploration) / cost
    # The penalty pushes away from known-good, explored areas.
    # The exploration bonus rewards uncertain regions.
    score = (ei - penalty + 0.05 * explore_scale * explore) / cost_safe
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
