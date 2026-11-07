"""Rewritten acquisition for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: total wall-clock budget for the whole evaluator run is 330s.
# We track time inside acquisition to avoid pathological loops (though acquisition
# is vectorized and fast, this is a safety net per the directive).
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill

# Problem constants
_F_MIN = -39.1662
_BOUNDS = np.array([-5.0, 5.0])


def _safe_z(x):
    """Compute z = (f_best - mu) / sigma safely."""
    return x


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) based on the progress of optimization.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Return a deterministic fallback: prefer low mu
        return -mu.copy()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB
    # UCB = -mu + beta * sigma (we minimize, so we want low mu and high uncertainty)
    # The beta parameter controls exploration-exploitation tradeoff
    
    # Adaptive strategy:
    # - Early in optimization (f_best is far from f_min), use more exploration (higher beta)
    # - Later in optimization (f_best is close to f_min), use more exploitation (lower beta)
    
    # Progress metric: how close are we to the known minimum?
    # f_best starts high (e.g., ~40-100) and decreases toward -39.1662
    # Define progress as normalized distance from initial estimate to f_min
    # A simple heuristic: use f_best relative to f_min
    
    # Scale: when f_best is much larger than f_min, we're early
    # When f_best is close to f_min, we're late
    # Use a sigmoid-like transition
    
    # Typical f_best range: from ~50-100 down to ~-39
    # Let's define a reference scale
    f_min = _F_MIN
    # Initial f_best is typically around 20-50 for ST1D with random starts
    # Use a smooth transition based on f_best
    
    # Define beta as a function of f_best
    # When f_best is high (early), beta is large (more exploration)
    # When f_best is low (late), beta is small (more exploitation)
    
    # Use a logistic function for smooth transition
    # Transition center: midpoint between typical initial and final f_best
    # Initial ~30, final ~-39, midpoint ~ -4.5
    # Scale: width of transition ~20
    
    center = -4.5
    width = 20.0
    
    # beta ranges from beta_max (early) to beta_min (late)
    beta_max = 3.0
    beta_min = 0.5
    
    # Logistic: t = 1 / (1 + exp(-(f_best - center) / width))
    # When f_best >> center, t ~ 1, beta ~ beta_max
    # When f_best << center, t ~ 0, beta ~ beta_min
    t = 1.0 / (1.0 + np.exp(-(f_best - center) / width))
    beta = beta_min + t * (beta_max - beta_min)
    
    # UCB score (higher is better for acquisition)
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is always >= 0, UCB can be negative
    # We want to blend them such that both contribute meaningfully
    
    # Option 1: Simple weighted sum with adaptive weight
    # Option 2: Use EI when close to optimum, UCB when far
    
    # Let's use a weighted sum where the weight shifts from UCB to EI as we progress
    # w_ei increases as we get closer to the optimum
    
    # w_ei = 1 - t (so w_ei is high when t is low, i.e., late in optimization)
    w_ei = 1.0 - t
    w_ucb = t
    
    # To make EI and UCB comparable, scale them
    # EI max value is roughly f_best - f_min (the maximum possible improvement)
    # UCB range depends on mu and sigma
    
    # Simple approach: just blend the raw scores
    # But EI is in objective units (improvement), UCB is in objective units (predicted value + exploration)
    
    # Actually, let's think about this differently.
    # EI gives the expected improvement over f_best
    # UCB gives a score that balances low predicted value and high uncertainty
    
    # A common approach: use EI for exploitation and add an exploration term
    # Or: use a single acquisition with adaptive parameter
    
    # Let's try: score = w_ei * EI + w_ucb * (UCB normalized)
    
    # Normalize UCB to be on a similar scale as EI
    # EI is typically in the range [0, f_best - f_min]
    # UCB is in the range of mu values plus beta*sigma
    
    # For simplicity, let's just use EI with an adaptive beta-like term
    # Or better: use the standard approach of blending
    
    # Actually, the simplest effective approach:
    # Use EI when we're close to the optimum, and UCB when we're far
    # Blend smoothly
    
    # Scale EI to be comparable to UCB
    # EI_max_possible = f_best - f_min (theoretical max improvement)
    ei_scale = max(f_best - f_min, 1e-6)
    ei_normalized = ei / ei_scale  # Now in [0, ~1]
    
    # UCB: normalize to be in a similar range
    # mu is in objective units, sigma is in objective units
    # Let's just use the raw UCB and EI, and let the weights handle the scale
    # But this may not work well if the scales differ greatly
    
    # Alternative: use EI directly, and add a UCB-like exploration bonus
    # score = EI + gamma * sigma
    # where gamma is adaptive
    
    # Let's try this simpler approach:
    # score = EI + beta * sigma
    # This is like EI with an additional exploration term
    
    # Actually, let's stick with the weighted blend but ensure the scales are reasonable
    # Use EI as the primary score and UCB as secondary
    
    # Final approach: weighted sum of EI and UCB, with EI scaled by its max possible value
    # and UCB left as is, then normalize both to [0, 1] range for blending
    
    # Compute min and max for normalization
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
