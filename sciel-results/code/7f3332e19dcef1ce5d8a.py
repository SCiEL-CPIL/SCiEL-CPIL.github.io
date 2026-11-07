"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI)
    with an exploration bonus based on uncertainty, plus a small
    quantile-based diversification term to prevent clustering.
    """
    # Time guard: if we've been running too long, fall back to simple EI
    # (though acquisition itself is vectorized and fast)
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical edge cases where EI might be nan or inf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Exploration term: LCB-like but scaled
    # We want to balance exploitation (EI) and exploration (uncertainty)
    # A common approach is to add a weighted sigma term
    
    # Compute a normalized uncertainty score
    # Use the percentile of sigma to normalize
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-10:
        sigma_median = 1e-10
    
    # Exploration bonus: encourages points with high sigma
    # But we want to avoid pure exploration; EI already has some exploration
    # Let's add a small multiple of sigma to encourage exploration of high-variance regions
    # The coefficient 0.5 is a heuristic balance
    
    # Also add a small quantile-based diversification:
    # Partition candidates by mu quantiles and add a bonus for being in
    # under-explored quantile bins (approximated by sigma within bin)
    
    # Simpler robust approach: combine EI with a scaled sigma term
    # score = EI + beta * sigma
    # But EI already correlates with sigma. Let's use a different blend.
    
    # Alternative: Use a modified EI that has a stronger exploration component
    # by adjusting the exploitation/exploration tradeoff.
    
    # Let's try: score = EI + 0.1 * sigma * (1 - norm.cdf(z))
    # This adds extra weight to exploration when improvement is uncertain.
    
    # Actually, let's keep it simple and effective:
    # Use EI as the primary signal, but add a small exploration bonus
    # proportional to sigma, weighted by how "uncertain" the improvement is.
    
    # The term (1 - norm.cdf(z)) is the probability that f(x) > f_best
    # i.e., the probability of NOT improving. We want to explore where this is high
    # but also where the potential improvement is large.
    
    # A well-known alternative is the Lower Confidence Bound (LCB):
    # lcb = mu - kappa * sigma
    # But since we're minimizing, lower is better, so LCB = mu - kappa * sigma
    # However, the task says "higher meaning evaluate next", so we need to
    # return scores where higher is better.
    
    # For minimization:
    # EI gives higher scores where expected improvement is larger.
    # LCB for minimization: we want low mu and high sigma.
    # So LCB_score = -mu + kappa * sigma (higher is better for selection)
    
    # Let's blend EI and a normalized LCB-like term.
    
    # Normalize EI to [0, 1] roughly
    ei_max = np.max(ei) if np.any(ei > 0) else 1e-10
    ei_min = np.min(ei) if np.any(ei < ei_max) else 0.0
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # LCB score: -mu + kappa * sigma
    # We need to normalize this too
    kappa = 1.0
    lcb_score = -mu + kappa * sigma_safe
    
    lcb_max = np.max(lcb_score)
    lcb_min = np.min(lcb_score)
    if lcb_max > lcb_min:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # Blend: 70% EI, 30% LCB
    # This should give a good balance of exploitation and exploration
    score = 0.7 * ei_norm + 0.3 * lcb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
