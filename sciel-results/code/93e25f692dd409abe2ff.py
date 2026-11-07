# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import erf

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin vs 330s kill

def _norm_cdf_vec(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def _norm_pdf_vec(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def _ei(mu, sigma, fb):
    s = np.maximum(sigma, 1e-8)
    z = (fb - mu) / s
    return (fb - mu) * _norm_cdf_vec(z) + s * _norm_pdf_vec(z)

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)

    # Portfolio of acquisition variants; return the one that best
    # concentrates search near plausible minima. We select based on a
    # cheap quality proxy computed on the scores themselves.
    s = np.maximum(sigma, 1e-8)

    cand = {}

    # 1) Pure Expected Improvement
    cand['ei'] = _ei(mu, s, fb)

    # 2) UCB family for MINIMIZATION: -mu + beta*sigma
    # Maximizing this score favors low mu (exploitation) and high sigma (exploration)
    cand['ucb1'] = -mu + 1.0 * s
    cand['ucb2'] = -mu + 2.0 * s
    cand['ucb05'] = -mu + 0.5 * s

    # 3) GP-Hype-like: EI but with the (fb-mu) term only (ignoring sigma pdf)
    z = (fb - mu) / s
    cand['hype'] = np.maximum(fb - mu, 0.0) * _norm_cdf_vec(z)

    # 4) Knowledge gradient proxy: EI / sigma  (rewards confident improvements)
    cand['kg'] = _ei(mu, s, fb) / s

    # 5) Probability of Improvement
    cand['pi'] = _norm_cdf_vec(z)

    # Select the variant whose top scores are sharpest (lowest top-1 mu)
    # while keeping finite, well-spread scores.
    best_name, best_scores = None, None
    best_key = float('inf')
    for name, sc in cand.items():
        if not np.all(np.isfinite(sc)):
            continue
        order = np.argsort(sc)[::-1]
        top10 = mu[order[:10]]
        # prefer variants whose top-10 mean mu is lowest (most focused)
        key = float(np.mean(top10))
        if key < best_key:
            best_key, best_name, best_scores = key, name, sc

    if best_scores is None:
        best_scores = cand['ei']

    # Normalize to a clean positive scale for downstream "higher = evaluate".
    # The evaluator only needs relative ordering, so keep as-is (finite).
    out = np.asarray(best_scores, dtype=float)
    out = np.where(np.isfinite(out), out, 0.0)
    return out
# EVOLVE-BLOCK-END
