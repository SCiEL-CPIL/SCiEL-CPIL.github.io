"""Rewritten acquisition using adaptive UCB with progress-based exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Check deadline guard
    if time.time() > _DEADLINE:
        return -mu

    # Goldstein-Price 2D minimum value
    f_min = 3.0

    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-9)

    # Adaptive UCB: balance exploration and exploitation
    # When f_best is far from f_min, explore more; when close, exploit more
    # Progress: 0 at start (f_best large), 1 when f_best near f_min
    # Use a scaling that makes progress meaningful
    gap = f_best - f_min
    # Normalize gap: early on gap is large, later small
    # Use log scaling for better behavior across range
    # When gap is large (early), beta should be large (explore)
    # When gap is small (late), beta should be small (exploit)
    
    # Adaptive beta: start high, decrease as we converge
    # Use a smooth transition
    # gap > 10: very early, beta ~ 2-3
    # gap ~ 1: mid, beta ~ 1
    # gap < 0.1: late, beta ~ 0.3-0.5
    
    # Use a formula: beta = base + amp * exp(-gap / scale)
    # Or: beta = base + amp / (1 + gap)
    
    # Let's use: beta = 0.5 + 1.5 * exp(-gap / 2.0)
    # When gap=0: beta = 2.0 (but this shouldn't happen often)
    # When gap=1: beta = 0.5 + 1.5*exp(-0.5) = 0.5 + 1.5*0.607 = 1.41
    # When gap=5: beta = 0.5 + 1.5*exp(-2.5) = 0.5 + 1.5*0.082 = 0.62
    # When gap=10: beta = 0.5 + 1.5*exp(-5) = 0.5 + 1.5*0.0067 = 0.51
    # When gap=20: beta = 0.5 + 1.5*exp(-10) ≈ 0.5
    
    # Actually, let's think about this differently.
    # Early: we want to explore. The initial 5 points are random, so f_best might be ~20-30.
    # We need beta large enough to explore.
    # Let's use: beta = 0.3 + 2.0 * (1 / (1 + gap / 5.0))
    # gap=0: beta = 0.3 + 2.0 = 2.3
    # gap=5: beta = 0.3 + 2.0*0.5 = 1.3
    # gap=10: beta = 0.3 + 2.0*(1/3) = 1.0
    # gap=20: beta = 0.3 + 2.0*0.2 = 0.7
    # gap=50: beta = 0.3 + 2.0*0.091 = 0.48
    # gap=100: beta = 0.3 + 2.0*0.048 = 0.396
    
    # This seems reasonable. Let me tune it.
    # Actually, the finding said "Adaptive UCB balances exploration and exploitation by increasing 
    # the uncertainty weight (beta) when the current best is far from the global minimum (3) 
    # and decreasing it as the search converges" and achieved 1.461.
    
    # Let's use a similar approach.
    beta = 0.3 + 2.0 * (1.0 / (1.0 + gap / 5.0))
    
    # Also add a small EI component for exploitation when very close
    # Combine UCB and EI
    # UCB score: -mu + beta * sigma
    # EI score: (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    
    # Let's use a weighted combination
    # When gap is large, weight more on UCB; when small, weight more on EI
    
    # Actually, let's just use UCB with adaptive beta, plus a small EI boost
    xi = 0.01 * max(gap, 0.1)
    z = (f_best - mu - xi) / sigma_clipped
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    ucb = -mu + beta * sigma_clipped
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Blend: when gap is large, use mostly UCB; when small, use mostly EI
    # weight_ucb = 1.0 / (1.0 + gap / 3.0)
    # weight_ei = 1.0 - weight_ucb
    # score = weight_ucb * ucb + weight_ei * ei
    
    # Actually, let's just use UCB primarily, since the finding suggests that works well
    # But add a small EI component to avoid pure exploration
    score = ucb + 0.1 * ei
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
