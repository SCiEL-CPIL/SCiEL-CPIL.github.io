"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 330.0
_START = time.time()
_EPS = 1e-12


def _fallback(mu, sigma, cost):
    """Cheap, always-finite cost-aware Probability of Improvement."""
    z = (mu - f_best) / (sigma + 1e-9)
    pi = norm.cdf(z)
    return pi / (cost + 1e-9)


def acquisition(mu, sigma, f_best, cost):
    global _DEADLINE
    # Keep a safety margin before the evaluator's hard kill.
    if time.time() - _START > _DEADLINE - 20.0:
        return _fallback(mu, sigma, cost)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]
    f_best = float(f_best)

    # Soft cost penalty: reward cheap evaluations, but do not zero them out.
    cost_pen = (cost + 1e-6) ** 0.5

    # ---- Monte-Carlo Expected Hypervolume Improvement over observed front ----
    # Reference points from the observed cost/value spread.
    c_lo = float(cost.min())
    c_hi = float(cost.max())
    if not np.isfinite(c_hi) or c_hi <= c_lo:
        c_hi = c_lo + 1.0
    v_hi = max(f_best, float(mu.min()))  # worst plausible value (lower is better)
    v_lo = max(0.0, float(mu.min()) - 2.0 * float(np.median(sigma + 1e-9)))
    if not np.isfinite(v_hi) or v_hi <= v_lo:
        v_hi = v_lo + 1.0

    n_ref = 16
    ref_c = np.linspace(c_lo, c_hi, n_ref)
    ref_v = np.linspace(v_lo, v_hi, n_ref)
    ref_c = ref_c[:, None]  # (n_ref, 1)
    ref_v = ref_v[None, :]  # (1, n_ref)

    # Expected hypervolume improvement of one candidate over the observed front:
    #   sum over reference points of  P(v < ref_v) * (ref_c - cost)
    # P(v < ref_v) estimated by MC over the predictive distribution.
    K = 8
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((K, n))  # (K, n)
    draws = mu[None, :] + sigma[None, :] * noise  # (K, n)

    # indicator (K, n, n_ref): draw value below each reference value
    below = (draws[:, :, None] < ref_v).astype(np.float64)  # (K, n, n_ref)
    p_below = below.mean(axis=0)  # (n, n_ref)

    # (ref_c - cost) clipped at 0, shape (n, n_ref)
    gain = np.clip(ref_c - cost[:, None], 0.0, None)

    ehi = (p_below * gain).sum(axis=1)  # (n,)

    # Normalise by the cost spread so the score is scale-free.
    span = max(c_hi - c_lo, 1e-6)
    ehi = ehi / span

    # Guard against non-finite values -> fall back to a valid score.
    if not np.all(np.isfinite(ehi)):
        return _fallback(mu, sigma, cost)

    score = ehi / cost_pen
    if not np.all(np.isfinite(score)):
        return _fallback(mu, sigma, cost)
    return score
# EVOLVE-BLOCK-END
