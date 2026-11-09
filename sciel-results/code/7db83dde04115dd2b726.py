"""Bayesian optimization acquisition for Hartmann 3D minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

_START = time.time()
_DEADLINE = _START + 325.0

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_INV_SQRT_2 = 1.0 / np.sqrt(2.0)


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Expected Improvement with UCB.
    Uses dynamic kappa that increases with number of evaluations (inferred from f_best history is not available,
    so we use a fixed reasonable kappa that balances exploration/exploitation for 30 evaluations).
    
    For minimization:
      EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
      UCB = mu - kappa * sigma  (we want low mu, so high score = -mu + kappa*sigma)
    
    We compute EI and a UCB-like score, normalize both to [0,1] range across the 2000 candidates,
    then blend: score = alpha * EI_norm + (1 - alpha) * UCB_norm
    
    This balances exploitation (EI) with exploration (UCB).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        scores = f_best - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)
    if n == 0:
        return np.array([])

    sigma_safe = np.maximum(sigma, 1e-8)
    delta = f_best - mu  # positive when mu < f_best

    z = delta / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)

    # EI computation
    phi_z = np.exp(-0.5 * z_clipped * z_clipped) / _SQRT_2PI
    Phi_z = 0.5 * (1.0 + erf(z_clipped * _INV_SQRT_2))
    ei = delta * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # UCB score for minimization: we want to select points with low mu and high sigma
    # score_ucb = -mu + kappa * sigma
    # We use a moderate kappa. For ~30 evaluations on 3D, kappa around 0.5-1.5 is reasonable.
    # Let's use kappa = 0.8 as a balanced choice
    kappa = 0.8
    ucb_score = -mu + kappa * sigma_safe
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)

    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min + 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones(n) * 0.5

    # Normalize UCB score to [0, 1]
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max > ucb_min + 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones(n) * 0.5

    # Blend: 60% EI, 40% UCB
    alpha = 0.6
    scores = alpha * ei_norm + (1.0 - alpha) * ucb_norm

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
