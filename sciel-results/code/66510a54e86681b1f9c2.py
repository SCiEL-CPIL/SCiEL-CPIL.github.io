"""Rewritten cost-aware acquisition for levy 2d BO."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 320.0


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.

    Structural change: instead of a single EI/cost scalar, we blend three
    complementary signals:
      1) Cost-adjusted Expected Improvement (EI / cost)
      2) Probability of Improvement (PI) discounted by cost
      3) A lower-tail risk signal: P(f(x) < f_best - margin) / cost
    This multi-criteria score better balances exploration vs exploitation
    under the non-uniform cost landscape, and the risk term actively
    pushes toward regions with high probability of significant improvement.
    """
    # Guard: if dangerously close to wall-clock limit, return safe deterministic signal
    if _elapsed() > _DEADLINE - 2.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1.0, None)  # cost is 1 + mean(x_scaled) so >= 1
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Standardized distance
    z = (f_best - mu) / sigma
    z = np.clip(z, -15.0, 15.0)

    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    # 1) Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)

    # 2) Probability of Improvement
    pi = cdf_z

    # 3) Risk-adjusted improvement: probability of beating f_best by a margin
    # Use a small margin relative to current uncertainty to focus on promising regions
    margin = 0.1 * sigma + 1e-6
    z_margin = (f_best - margin - mu) / sigma
    z_margin = np.clip(z_margin, -15.0, 15.0)
    risk_signal = norm.cdf(z_margin)

    # Cost-adjusted scores
    score_ei = ei / cost
    score_pi = pi / cost
    score_risk = risk_signal / cost

    # Blend: weight EI most heavily, PI second, risk as a tiebreaker/exploration booster
    # The risk term is scaled down so it doesn't dominate but helps break ties
    # and encourages slightly more aggressive exploration
    w_ei = 1.0
    w_pi = 0.3
    w_risk = 0.2

    score = w_ei * score_ei + w_pi * score_pi + w_risk * score_risk

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
