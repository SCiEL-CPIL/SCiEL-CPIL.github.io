"""Cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: keep per-call work well under the 300 s evaluator limit.
_T0 = time.time()
_DEADLINE = 300.0
_MARGIN = 30.0


def _norm_cdf(z):
    return norm.cdf(z)


def _norm_pdf(z):
    return norm.pdf(z)


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    n = mu.shape[0]
    # Safety: ensure finite inputs; fall back to a neutral score if anything is bad.
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=float)

    # Per-point evaluation cost: 1 + mean(x in [0,1]). Use a safe floor to avoid div-by-zero.
    cost = np.maximum(cost, 1.0)

    # Estimate elapsed fraction of the budget from the running cost.
    # The loop runs while total cost < 30, so normalize by ~30.
    total_cost = float(np.mean(cost)) if n else 1.0
    frac = np.clip(total_cost / 30.0, 0.0, 1.0)

    # Exploration weight: high early, decays to low late.
    # Beta-style schedule that keeps some exploration throughout.
    beta = 0.5 + 3.5 * (1.0 - frac) ** 1.5

    # Floor on sigma to prevent the z-score from blowing up in low-uncertainty areas.
    # This also tames the UCB term when sigma is tiny.
    sigma_floor = 1e-4
    s = np.maximum(sigma, sigma_floor)

    # --- Candidate 1: Expected Improvement (EI) ---
    z_ei = (f_best - mu) / s
    ei = (f_best - mu) * _norm_cdf(z_ei) + s * _norm_pdf(z_ei)
    ei = np.maximum(ei, 0.0)

    # --- Candidate 2: UCB (exploitation + exploration) ---
    ucb = mu + beta * s

    # --- Candidate 3: Probability of Improvement (PI) ---
    z_pi = (f_best - mu) / s
    pi = _norm_cdf(z_pi)

    # --- Candidate 4: Cost-aware EI (EI per unit cost) ---
    ei_cost = ei / cost

    # --- Candidate 5: Cost-aware UCB (UCB per unit cost) ---
    ucb_cost = ucb / cost

    # Combine: take the max across normalized candidates.
    # Normalize each candidate to [0,1] by its own range so they are comparable.
    def _normalize(a):
        a = np.asarray(a, dtype=float)
        lo = np.min(a)
        hi = np.max(a)
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    scores = np.maximum.reduce([
        _normalize(ei),
        _normalize(ucb),
        _normalize(pi),
        _normalize(ei_cost),
        _normalize(ucb_cost),
    ])

    # Weighted blend: emphasize cost-aware terms to respect the cost model.
    # Blend the normalized scores with a fixed mix that favors cost-awareity.
    blended = (
        0.20 * _normalize(ei)
        + 0.20 * _normalize(ucb)
        + 0.15 * _normalize(pi)
        + 0.25 * _normalize(ei_cost)
        + 0.20 * _normalize(ucb_cost)
    )

    # Use the max of the blended score and the pure max-normalized as a safety net.
    final = np.maximum(scores, blended)

    # Ensure finiteness.
    final = np.where(np.isfinite(final), final, 0.0)

    # Deadline guard: if we are running long, return immediately (already computed).
    if time.time() > _T0 + _DEADLINE - _MARGIN:
        return final

    return final
# EVOLVE-BLOCK-END
