"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach combining Expected Improvement (EI) and LCB:
    - EI handles exploitation of regions likely to improve upon f_best.
    - LCB handles exploration of high-uncertainty regions.
    - Adaptive blending: more exploration early, more exploitation later.
    
    The key structural idea from inspirations: use a pure LCB with adaptive
    exploration constant that starts high and decays as f_best improves.
    The refinement from the current program: blend with EI using a sigmoid
    adaptation based on progress toward the known optimum.
    """
    # Time guard: if we are close to the deadline, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Compute LCB for minimization
    # We want to minimize LCB = mu + k * sigma, so maximize -LCB = -mu - k * sigma
    # Adaptive exploration constant:
    # - Start with high k for exploration when f_best is far from optimum
    # - Decay k as f_best gets closer to the known minimum (-1.8013)
    
    min_val = -1.8013
    distance = f_best - min_val  # Always positive (f_best >= min_val)
    
    # Clip distance to avoid extreme values
    distance_clipped = np.clip(distance, 0, 1.0)
    
    # Adaptive k: high when far from optimum, low when close
    # k = k_min + (k_max - k_min) * (distance / distance_scale)
    distance_scale = 0.5
    k_min = 0.5
    k_max = 3.0
    
    k = k_min + (k_max - k_min) * (distance_clipped / distance_scale)
    k = np.clip(k, k_min, k_max)
    
    LCB_score = -(mu + k * sigma_safe)
    
    # Adaptive blending:
    # - Early on (f_best far from optimum), favor LCB (exploration)
    # - Later (f_best close to optimum), favor EI (exploitation)
    
    # Use sigmoid adaptation based on progress toward optimum
    # When f_best is close to -1.8013, alpha (EI weight) should be high
    # When f_best is far, alpha should be low
    
    # alpha = 0.5 + 0.5 * sigmoid((f_best + 1.5) / 0.3)
    # This gives alpha ~0.5 when f_best = -1.5, alpha ~0.9 when f_best = -1.6,
    # and alpha ~1.0 when f_best = -1.8
    
    alpha = 0.5 + 0.5 * (1.0 / (1.0 + np.exp(-((f_best + 1.4) / 0.2))))
    
    # Clamp alpha to maintain some exploration always
    alpha = np.clip(alpha, 0.2, 0.85)
    
    # Combine scores: weight EI and LCB
    # Normalize both to comparable scales
    # EI is in objective units, LCB_score is also in objective units
    
    score = alpha * EI + (1 - alpha) * LCB_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
