"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: EI with adaptive UCB exploration.
    
    Early in the search (few evaluations), we need strong exploration to find
    the basin of the global minimum. As we get more evaluations, we shift
    toward exploitation. We estimate the evaluation count from the spread of
    sigma: early on sigma is large everywhere; later it's small near evaluated
    points.
    
    We use a weighted combination of Expected Improvement and a UCB-style
    term. The exploration weight decays as the median sigma decreases.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # EI component
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB component: mu + kappa * sigma (we want to minimize, so we prefer
    # low mu but high sigma for exploration). For a minimization UCB we use
    # -mu + kappa*sigma, but since we're combining with EI (which already
    # favors low mu), let's use a pure exploration bonus: kappa * sigma.
    #
    # Adapt kappa based on how much uncertainty remains.
    # Early: median sigma is large -> explore more.
    # Late: median sigma is small -> exploit more.
    med_sigma = np.median(sigma)
    
    # Normalize: if med_sigma is large, we're early; if small, we're late.
    # For branin on [-5,10]x[0,15] with length scale 0.2 on [0,1] scaled coords,
    # the raw sigma in objective units starts around the target std and shrinks.
    # Use a soft schedule: kappa starts high and decays.
    #
    # We don't know the exact eval count, but sigma median is a good proxy.
    # Let's use: kappa = base_kappa * (med_sigma / (med_sigma + scale))
    # where scale is a characteristic sigma value.
    #
    # A simpler robust approach: use a fixed moderate kappa that works well
    # for the typical trajectory. From experience with 2D branin and ~25 evals,
    # kappa around 1.5-2.5 on the standardized scale works, but our sigma is
    # in objective units. Let's estimate the target std from the spread of mu.
    
    # Estimate the objective scale from the data
    mu_range = np.ptp(mu)
    if mu_range < 1e-9:
        obj_scale = 1.0
    else:
        obj_scale = max(mu_range / 4.0, 1e-6)
    
    # Adaptive kappa: larger when uncertainty is high relative to objective scale
    # kappa in "sigma units" — we add kappa * sigma to a UCB-like score
    # For minimization, UCB score = mu - kappa*sigma (we want to pick low values)
    # But we're returning "higher = evaluate next", so we negate: -mu + kappa*sigma
    # However, combining with EI: let's just add an exploration bonus to EI.
    
    # Exploration bonus: favors points with high sigma
    # Scale it so it's meaningful relative to EI magnitudes
    ei_scale = np.max(np.abs(ei)) if np.max(np.abs(ei)) > 1e-9 else 1.0
    
    # Use a schedule: early (high med_sigma) -> more exploration
    # med_sigma / obj_scale gives a dimensionless uncertainty measure
    unc_ratio = med_sigma / obj_scale
    
    # kappa: when unc_ratio is large (early), explore more
    # Clamp to a reasonable range
    kappa = 0.5 + 2.0 * np.exp(-unc_ratio)
    
    # Exploration bonus in EI units
    exp_bonus = kappa * (sigma / obj_scale) * ei_scale
    
    # Combine: EI + exploration bonus
    scores = ei + exp_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
