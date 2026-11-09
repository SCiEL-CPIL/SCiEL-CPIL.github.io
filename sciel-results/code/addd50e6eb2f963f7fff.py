"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5 second safety margin before 330s kill


def _ei(mu, sigma, f_best):
    """Standard Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, f_best, beta):
    """Upper Confidence Bound (for minimization: lower is better)."""
    return mu - beta * sigma


def _qei(mu, sigma, f_best, q=0.5):
    """Quantile Expected Improvement: EI at quantile q of the improvement distribution.
    For minimization, improvement = f_best - f(x), so we want high values.
    The q-th quantile of improvement: f_best - mu - z_q * sigma where z_q is the (1-q) quantile of N(0,1)
    Actually, let's think more carefully.
    
    Improvement I = f_best - f(x) where f(x) ~ N(mu, sigma^2)
    I ~ N(f_best - mu, sigma^2)
    The expected improvement E[I^+] where I^+ = max(I, 0) is standard EI.
    
    For a "quantile" approach, we can use the median of the positive part, or 
    we can use a different formulation. Let's use a simpler structural change:
    Use a combination of EI and a penalty for high uncertainty in regions far from f_best.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid approach:
    - Early on (when we've only seen a few points), use more exploration (UCB-like)
    - Later, use exploitation (EI-like)
    
    The blend is based on the number of distinct observations inferred from sigma patterns,
    but since we don't have that info directly, we use a time-based or f_best-based heuristic.
    
    Structural change: Use a "noisy EI" variant that accounts for the fact that in early
    iterations, the GP is poorly calibrated, so we add an exploration bonus that scales
    with the global sigma range.
    """
    # Time guard
    if time.time() > _DEADLINE:
        # Fallback: return EI scores to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute standard EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB for exploration
    # Use a beta that adapts: larger when sigma range is large (early, uncertain)
    sigma_range = np.ptp(sigma)
    if sigma_range < 1e-6:
        # All sigmas similar, just use EI
        return ei
    
    # Normalize sigma to [0, 1] for adaptive beta
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min + 1e-9)
    
    # Beta scales with the overall uncertainty level
    # When sigma_range is large relative to f_best, we're in early exploration
    # Use a structural blend: weight = 1 / (1 + sigma_range / max(f_best, 1e-3))
    # This gives more exploration weight when sigma_range is large compared to f_best
    
    # Alternative structural approach: Use a "thompson sampling" inspired score
    # Sample from the posterior and take the minimum (for minimization)
    # But we can't sample 2000 points efficiently per call... 
    # Actually we can: it's just 2000 samples.
    
    # Let's try a different structural approach: 
    # Use the "probabilistic regret" formulation:
    # P(f(x) < f_best) * (f_best - mu) + P(f(x) > f_best) * sigma * pdf(z)
    # This is essentially EI but let's try a different weighting.
    
    # Structural change: Use a "sharp" EI that emphasizes regions where improvement is likely
    # by using a higher-order moment or a different kernel.
    
    # Actually, let's try a simpler but structurally different approach:
    # Use the "lower confidence bound" with an adaptive kappa based on the 
    # ratio of best to mean mu.
    
    # Compute the ratio of f_best to the mean of mu
    mu_mean = np.mean(mu)
    if mu_mean > 0:
        ratio = f_best / mu_mean
    else:
        ratio = 1.0
    
    # When f_best is close to mu_mean, we're exploiting; when far, exploring
    # Use a smooth transition
    # kappa = 1 + ratio (higher kappa = more exploration)
    # But this is similar to LCB...
    
    # Let's try: Use EI but with a "temperature" that sharpens the selection
    # in regions where EI is high.
    
    # Actually, the most structurally different thing I can do is use
    # a "batch" aware acquisition that considers the diversity of candidates.
    # But we only get mu and sigma, not the actual points.
    
    # Let me try: Use the "expected hypervolume improvement" approximation
    # For 1D improvement, this reduces to EI, but we can add a penalty for
    # candidates that are similar to already-selected ones. But we don't have
    # the selected points...
    
    # OK, let's try a different structural approach:
    # Use a "noisy" version of EI where we add a small random perturbation
    # to encourage exploration. But this is stochastic and may not help.
    
    # Let me try: Use the "probability of improvement" (PI) instead of EI.
    # PI = P(f(x) < f_best) = norm.cdf(z)
    # This is more exploratory than EI because it doesn't weight by the amount of improvement.
    
    pi = norm.cdf(z)
    
    # Blend PI and EI based on how "confident" we are in f_best
    # Confidence is high when sigma is small at the best point (but we don't know which is best)
    # Use the minimum sigma as a proxy for confidence
    min_sigma = np.min(sigma)
    confidence = 1.0 / (1.0 + min_sigma / max(abs(f_best), 1e-3))
    
    # Blend: high confidence -> more EI, low confidence -> more PI
    score = confidence * ei + (1 - confidence) * pi
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
