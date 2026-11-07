"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0

    # Ensure arrays are 1-D numpy arrays of the right type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    if n == 0:
        return np.array([], dtype=np.float64)

    # Floor sigma to avoid division by zero / log of zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # Potential improvement (we are minimizing, so improvement = f_best - mu)
    delta = f_best - mu

    # ---------- Expected Improvement (EI) ----------
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # ---------- Hypervolume-style Expected Improvement (HEI) ----------
    # HEI rewards larger improvements more than linear EI via a log bonus.
    # This helps distinguish between marginal and substantial improvements,
    # which is crucial for multi-modal functions like rastrigin and hartmann.
    hei = ei * (1.0 + np.log1p(ei / (sigma_safe + 1e-30)))
    hei = np.maximum(hei, 0.0)

    # ---------- Thompson Sampling decision process ----------
    # Sample a single function value y ~ N(mu, sigma^2) for each candidate,
    # then select the candidate that minimizes y / cost(x), thereby naturally
    # balancing the stochastic trade-off between predicted objective improvement
    # and acquisition cost without requiring explicit exploration-exploitation.
    #
    # We approximate the expected value of the Thompson Sampling decision by
    # computing E[min(y / cost)] over the 2000 candidates. Since we can only
    # return one score per candidate, we use a proxy: the expected improvement
    # under Thompson Sampling is related to the probability that a candidate
    # would be selected, which is higher when y/cost is low.
    #
    # A practical approximation: use the quantile of y/cost distribution.
    # Candidates with low y/cost have high selection probability.
    # We compute a score proportional to the "expected regret reduction" from
    # Thompson Sampling, which is E[f_best - min(y/cost * cost)] = E[f_best - min(y)].
    # But since cost varies, we use: score_i = P(candidate i is the argmin of y/cost)
    # which is approximated by the inverse of the local density of y/cost values.
    #
    # Simpler and more effective: use the expected improvement from Thompson
    # Sampling, which for a single sample is E[max(0, f_best - y)] = EI (same as above).
    # But the key difference is that Thompson Sampling naturally accounts for
    # the cost by selecting based on y/cost, not just y.
    #
    # We approximate this by scaling EI by the inverse of the expected cost
    # of being selected. A candidate with lower cost has higher selection
    # probability, so we boost its score.
    #
    # The expected number of times candidate i is selected is proportional to
    # P(y_i / cost_i < y_j / cost_j for all j != i).
    # For simplicity, we use a rank-based approximation:
    #   ts_score_i = EI_i * (1 / rank(y_i / cost_i))
    # where rank is the ascending rank of y_i / cost_i.
    #
    # However, computing exact ranks is O(n log n) and may not be the best
    # use of time. Instead, we use a smooth approximation:
    #   ts_score_i = EI_i * exp(-beta * (y_i / cost_i - min(y/cost)))
    # where beta controls the sharpness of the selection.
    #
    # Since we don't have y (we only have mu and sigma), we use the expected
    # value of y/cost: E[y/cost] = mu / cost.
    # So we compute:
    #   ts_score_i = EI_i * exp(-beta * (mu_i / cost_i - min(mu/cost)))

    cost_safe = np.maximum(cost, 1e-12)

    # Compute mu/cost as a proxy for the expected y/cost
    mu_over_cost = mu / cost_safe
    min_mu_over_cost = np.min(mu_over_cost)

    # Beta controls the sharpness of Thompson Sampling selection
    # Higher beta means more aggressive selection of the best y/cost
    beta = 2.0

    # Thompson Sampling score: EI weighted by selection probability
    ts_scores = ei * np.exp(-beta * (mu_over_cost - min_mu_over_cost))
    ts_scores = np.maximum(ts_scores, 0.0)

    # ---------- Cost-aware scaling ----------
    # cost = 1 + mean(x_scaled) in [1, 2]. Prefer cheap points.
    # Adaptive alpha: early on we can afford to be less cost-sensitive;
    # later we tighten cost awareness to conserve budget.
    mean_cost = float(np.mean(cost))
    norm_cost = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)
    alpha = 0.35 + 0.55 * norm_cost  # 0.35 -> 0.90
    cost_weight = cost_safe ** (-alpha)

    # ---------- Exploration term (UCB) ----------
    # UCB for minimisation: f_best - mu + kappa * sigma
    mu_spread = np.std(mu) + 1e-30
    sigma_spread = np.std(sigma_safe)
    uncertainty_ratio = sigma_spread / (mu_spread + sigma_spread)
    kappa_base = 1.1
    kappa = kappa_base * (0.5 + 1.5 * uncertainty_ratio)

    ucb = f_best - mu + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)

    # ---------- Combine EI, HEI, TS and UCB with adaptive weights ----------
    def _normalise(arr):
        m = np.max(arr)
        if m > 1e-18:
            return arr / m
        return arr

    ei_n = _normalise(ei)
    hei_n = _normalise(hei)
    ts_n = _normalise(ts_scores)
    ucb_n = _normalise(ucb)

    # Weights: as the budget is consumed (norm_cost up), shift weight from
    # exploration (UCB) to exploitation (EI/HEI/TS).
    # TS gets a significant share because it naturally balances exploration
    # and exploitation through the stochastic selection process.
    w_ei = 0.20 + 0.15 * norm_cost      # 0.20 -> 0.35
    w_hei = 0.30 + 0.20 * norm_cost     # 0.30 -> 0.50
    w_ts = 0.25 + 0.15 * norm_cost      # 0.25 -> 0.40
    w_ucb = 1.0 - w_ei - w_hei - w_ts   # 0.25 -> -0.25 (clamped to 0)
    w_ucb = max(w_ucb, 0.0)

    # Renormalize weights to sum to 1
    w_sum = w_ei + w_hei + w_ts + w_ucb
    if w_sum > 1e-12:
        w_ei /= w_sum
        w_hei /= w_sum
        w_ts /= w_sum
        w_ucb /= w_sum

    scores = w_ei * ei_n + w_hei * hei_n + w_ts * ts_n + w_ucb * ucb_n

    # Apply cost weighting
    scores = scores * cost_weight

    # ---------- Final normalisation for numerical stability ----------
    s_mean = np.mean(scores)
    s_std = np.std(scores)
    if s_std > 1e-12:
        scores = (scores - s_mean) / s_std
    else:
        scores = scores - s_mean

    # Guarantee finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return np.zeros_like(mu)

    return scores
# EVOLVE-BLOCK-END
