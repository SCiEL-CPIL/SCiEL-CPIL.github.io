"""Cost-aware acquisition for Bayesian optimization (hartmann 6d).
SCiEL evolves the EVOLVE-BLOCK below."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def _softmax_scores(raw, temperature, n_explore, n_total):
    """Concentrate score mass on the top frontier of candidates.

    raw: (n_total,) higher = better raw acquisition.
    Returns (n_total,) finite scores in [0, 1] (sum ~1).
    """
    if n_total <= 0:
        return np.zeros(0, dtype=float)
    if not np.all(np.isfinite(raw)):
        raw = np.where(np.isfinite(raw), raw, -np.inf)
    if not np.any(np.isfinite(raw)):
        return np.full(n_total, 1.0 / n_total, dtype=float)

    # Pick the top candidates to form the "exploit frontier".
    k = max(1, min(n_explore, n_total))
    # argsort descending; take first k. Use a stable partition for speed.
    # np.argpartition is O(n); fine for 2000.
    top_idx = np.argpartition(raw, -k)[-k:]
    top_raw = raw[top_idx]
    top_finite = np.isfinite(top_raw)
    if not np.any(top_finite):
        return np.full(n_total, 1.0 / n_total, dtype=float)

    # Softmax over the top frontier with a temperature.
    # Normalize by the max for numerical stability.
    m = np.max(top_raw[top_finite])
    # For non-finite entries within top, push them to the bottom.
    safe_top = np.where(top_finite, top_raw, -np.inf)
    exps = np.exp((safe_top - m) / max(temperature, 1e-9))
    # Zero out non-finite contributions.
    exps = np.where(top_finite, exps, 0.0)
    s = np.sum(exps)
    if not np.isfinite(s) or s <= 0:
        # Uniform over the finite top entries.
        scores = np.zeros(n_total, dtype=float)
        scores[top_idx[top_finite]] = 1.0 / np.sum(top_finite)
        return scores
    scores = np.zeros(n_total, dtype=float)
    scores[top_idx] = exps / s
    return scores


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware expected-improvement acquisition with frontier concentration.

    mu, sigma, cost: (n,) arrays, n=2000.
    Returns (n,) finite scores, higher = evaluate next.
    """
    t0 = time.time()
    DEADLINE = t0 + 290.0  # safety margin vs 300s/330s limits

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = mu.shape[0]

    # Sanitize inputs.
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Expected improvement (standard form, numerically stable via log where possible).
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu)/sigma
    # For large positive z, EI ~ f_best - mu; for large negative z, EI ~ 0.
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values that make pdf underflow/overflow issues.
    z_clip = np.clip(z, -10.0, 10.0)
    cdf = norm.cdf(z_clip)
    pdf = norm.pdf(z_clip)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.where(ei > 0, ei, 0.0)

    # Cost-aware: divide by cost.
    raw = ei / cost

    # Optional: mild UCB-style uncertainty bonus to encourage exploration early.
    # We keep it small so exploitation dominates but we don't get stuck.
    # Scale the bonus by 1/sqrt(1 + log(1 + f_best - min_mu)) is complex; use a fixed small weight.
    # Instead, we rely on the frontier concentration for exploration balance.

    if not np.all(np.isfinite(raw)):
        raw = np.where(np.isfinite(raw), raw, 0.0)

    # Frontier concentration: how many top candidates to consider.
    # Start with a moderately small frontier, grow slowly as we see more data.
    # We don't have explicit iteration count, but we can infer from f_best vs initial.
    # Use a fixed, smallish frontier that adapts via temperature instead.
    n_explore = 32
    temperature = 0.15  # lower = more concentrated on the very top

    scores = _softmax_scores(raw, temperature, n_explore, n)

    # Deadline guard: if we're running long, just return what we have.
    if time.time() > DEADLINE:
        # Ensure finite and return.
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure all scores are finite and non-negative.
    scores = np.where(np.isfinite(scores) & (scores >= 0), scores, 0.0)

    # Normalize so max is 1 (the loop picks argmax, so scale doesn't matter,
    # but keeping bounded is safe).
    mx = np.max(scores)
    if mx > 0 and np.isfinite(mx):
        scores = scores / mx

    return scores
# EVOLVE-BLOCK-END
