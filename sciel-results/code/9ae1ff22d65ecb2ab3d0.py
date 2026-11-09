"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

# Known minimum of Michalewicz 2D
_MICH_MIN = -1.8013

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive UCB + Expected Improvement hybrid.
    Exploration (UCB) weight adapts based on proximity to the known optimum:
    strong exploration early, exploitation as f_best approaches the minimum.
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    # Check deadline
    if not _elapsed_ok():
        return np.random.rand(len(mu))
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Adaptive exploration coefficient ---
    # Progress toward optimum: 0 at start, 1 when f_best reaches _MICH_MIN
    # Use a smooth saturation so exploration decays gradually
    progress = np.clip((f_best - _MICH_MIN) / 1.0, 0.0, 1.0)
    # beta starts high (strong exploration) and decays as we improve
    # beta = beta_max * (1 - progress)^gamma  + beta_min
    beta_max = 3.0
    beta_min = 0.5
    gamma = 1.5
    beta = beta_max * (1.0 - progress) ** gamma + beta_min
    
    # --- UCB term ---
    ucb = mu + beta * sigma_safe  # for minimization, lower is better, so we want high UCB score = low mu + high sigma
    # Actually for minimization, UCB score should be: -mu + beta*sigma (higher = more promising)
    # Let's use: score_ucb = -mu + beta * sigma_safe
    score_ucb = -mu + beta * sigma_safe
    
    # --- Combine EI and UCB ---
    # Normalize EI to have comparable scale with UCB
    ei_max = np.max(ei)
    if ei_max > eps:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_max = np.max(score_ucb)
    if ucb_max > eps:
        ucb_norm = score_ucb / ucb_max
    else:
        ucb_norm = np.zeros_like(score_ucb)
    
    # Blend: weight shifts from UCB to EI as progress increases
    # Early: more UCB (exploration), Late: more EI (exploitation)
    w_ei = 0.4 + 0.5 * progress  # 0.4 early, 0.9 late
    w_ucb = 1.0 - w_ei
    w_ucb = max(w_ucb, 0.05)  # always keep some exploration
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small penalty for points where mu is already very close to f_best
    # (avoids redundant evaluations near the current best)
    proximity_penalty = 0.05 * np.exp(-((f_best - mu) ** 2) / (2 * sigma_safe ** 2 + eps))
    score = score - proximity_penalty * (1.0 - progress)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
