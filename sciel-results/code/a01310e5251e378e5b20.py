"""Cost-aware acquisition for griewank-2d Bayesian optimization (fresh design)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Safety deadline: evaluator kills at 330 s wall-clock; return safely early.
_DEADLINE = time.time() + 280.0


def _safe_z(mu, sigma, f_best):
    # Numerically-safe standardised improvement for minimisation.
    s = np.maximum(sigma, 1e-10)
    diff = f_best - mu
    # Guard: if sigma is tiny but diff positive, EI ~ diff; avoid overflow in pdf.
    z = diff / s
    # Clamp to keep cdf/pdf well-conditioned.
    z = np.clip(z, -40.0, 40.0)
    return z, s, diff


def acquisition(mu, sigma, f_best, cost):
    start = time.time()
    # Coerce inputs to 1-D float arrays.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]

    # --- Term 1: Expected Improvement (minimisation), per unit cost. ---
    z, s, diff = _safe_z(mu, sigma, f_best)
    # EI = diff * Phi(z) + s * phi(z); clip to non-negative.
    ei = diff * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Term 2: Posterior uncertainty (sigma) per unit cost, scaled. ---
    # Standardised uncertainty relative to current best-gap to keep magnitudes sane.
    sigma_term = s / (abs(f_best) + 1.0)

    # --- Term 3: Distance-to-current-best (in objective scale) via mu. ---
    # Reward regions where the GP mean is already low (exploit); this is -mu per cost.
    exploit_term = -mu / (abs(f_best) + 1.0)

    # Cost normalisation: penalise high-cost points. cost in [1,2] for griewank 2d.
    inv_cost = 1.0 / np.maximum(cost, 1e-9)

    # Blend: EI dominates, with a moderate exploration (sigma) and exploit (mu) tilt.
    w_ei = 1.0
    w_sigma = 0.35
    w_exploit = 0.5

    score = (w_ei * ei + w_sigma * sigma_term + w_exploit * exploit_term) * inv_cost

    # Guard against non-finite values.
    score = np.where(np.isfinite(score), score, 0.0)

    # --- Time guard: if we are close to the deadline, return the current
    #     (already-computed) scores immediately; the computation above is O(n)
    #     and finishes well within budget for n=2000. ---
    if time.time() > _DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
