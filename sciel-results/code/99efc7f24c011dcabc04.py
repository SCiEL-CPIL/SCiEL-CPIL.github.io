"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition: adaptive UCB blended with EI, normalized by cost."""
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple cost-normalized negative mu
        return -mu / cost

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # z-score for EI
    z = (f_best - mu) / sigma

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB with adaptive beta
    # Estimate progress: f_best relative to sigma scale
    # Early: more exploration; Late: more exploitation
    # Use ratio of f_best to typical sigma to gauge convergence
    sigma_mean = float(np.mean(sigma))
    # If f_best is large relative to sigma, we're early (need exploration)
    # If f_best is small, we're late (exploit)
    # Adaptive beta: start high, decrease as we converge
    # Normalize: use f_best / (sigma_mean + eps) as a progress indicator
    progress_ratio = abs(f_best) / (sigma_mean + 1e-8)
    # Clip progress ratio to [0, 10] for stability
    progress_ratio = np.clip(progress_ratio, 0, 10)
    # beta starts around 2.5 when progress_ratio is small, drops to ~0.5 when large
    # Linear interpolation
    beta = np.clip(2.5 - 0.2 * progress_ratio, 0.3, 2.5)

    ucb = -mu + beta * sigma

    # Blend: weight EI more when we're confident (small sigma), UCB when uncertain
    # Use sigma as a confidence indicator
    # Where sigma is small, EI is reliable; where sigma is large, UCB explores
    # Create a soft blend based on sigma relative to its own scale
    sigma_scale = sigma_mean + 1e-8
    # weight for EI: higher when sigma is low (confident)
    ei_weight = 1.0 / (1.0 + sigma / sigma_scale)
    # weight for UCB: higher when sigma is high (uncertain)
    ucb_weight = 1.0 - ei_weight

    # Combine
    raw = ei_weight * ei + ucb_weight * ucb

    # Cost normalization
    score = raw / cost

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
