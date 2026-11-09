"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement + uncertainty-driven exploration.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.clip(ei, -1e6, 1e6)

    # Scale reference: use f_best magnitude but floor at 1.0 for stability
    scale = max(abs(f_best), 1.0)

    # Relative uncertainty (how much we don't know relative to current best)
    rel_uncert = sigma_safe / scale

    # Cap exploration to prevent domination
    expl_bonus = np.minimum(rel_uncert, 5.0)

    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)

    # Normalize exploration to [0, 1]
    expl_max = np.max(expl_bonus)
    expl_min = np.min(expl_bonus)
    expl_range = expl_max - expl_min
    if expl_range > 1e-12:
        expl_norm = (expl_bonus - expl_min) / expl_range
    else:
        expl_norm = np.zeros_like(expl_bonus)

    # Blend with a slight emphasis on EI (exploitation) but keep meaningful exploration
    # 0.8/0.2 split: Beale has a narrow valley; too much exploration wastes evals
    w_ei = 0.80
    w_expl = 0.20
    scores = w_ei * ei_norm + w_expl * expl_norm

    # Mild penalty for candidates whose mean is well above f_best
    # Prevents selecting obviously poor regions even if they have high uncertainty
    mu_penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 3.0)
    scores = scores - 0.05 * mu_penalty

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Deadline guard (should be fast; this is just a safety net)
    if time.time() - t0 > 325.0:
        pass  # already computed; return immediately

    return scores
# EVOLVE-BLOCK-END
