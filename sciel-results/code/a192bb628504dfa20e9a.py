"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
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

    # ---------- Cost-aware scaling ----------
    # cost = 1 + mean(x_scaled) ∈ [1, 2].  We want to prefer cheap points.
    cost_safe = np.maximum(cost, 1e-12)

    # Adaptive alpha: early on we can afford to be less cost-sensitive
    mean_cost = float(np.mean(cost))
    norm_cost = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)
    alpha = 0.4 + 0.6 * norm_cost
    cost_weight = cost_safe ** (-alpha)

    # ---------- Student-t based Expected Improvement ----------
    # Use a Student-t approximation to account for model uncertainty more robustly.
    # Degrees of freedom: lower when uncertainty is high (more exploration),
    # higher when uncertainty is low (more exploitation).
    sigma_mean = float(np.mean(sigma_safe))
    sigma_std = float(np.std(sigma_safe))
    
    # Adapt degrees of freedom: more uncertainty -> lower df -> heavier tails
    # df in [3, 30]: lower df means more robust to outliers in improvement
    uncertainty_level = sigma_std / (sigma_mean + 1e-12)
    df = 3.0 + 27.0 * (1.0 - np.clip(uncertainty_level, 0.0, 1.0))
    
    # For Student-t with df degrees of freedom, the expected improvement
    # can be approximated. For large df it approaches normal EI.
    # We use a modified EI that accounts for the heavier tails.
    # The key idea: with Student-t, the effective "scale" is larger,
    # so we adjust the EI calculation.
    
    # Approximate Student-t EI using the fact that t_df has variance df/(df-2)
    # for df > 2. We scale sigma by sqrt(df/(df-2)) to account for this.
    if df > 2:
        scale_factor = np.sqrt(df / (df - 2.0))
    else:
        scale_factor = 2.0  # conservative for very low df
    
    sigma_t = sigma_safe * scale_factor
    z_t = delta / sigma_t
    cdf_t = norm.cdf(z_t)
    pdf_t = norm.pdf(z_t)
    ei_t = delta * cdf_t + sigma_t * pdf_t
    ei_t = np.maximum(ei_t, 0.0)

    # Blend between standard EI and Student-t EI based on uncertainty
    # When uncertainty is high, rely more on the robust Student-t version
    blend = np.clip(uncertainty_level, 0.0, 1.0)
    ei_blended = (1.0 - 0.5 * blend) * ei + 0.5 * blend * ei_t
    ei_blended = np.maximum(ei_blended, 0.0)

    # ---------- Exploration term (cost-aware UCB) ----------
    mu_spread = np.std(mu) + 1e-30
    sigma_spread = np.std(sigma_safe)
    uncertainty_ratio = sigma_spread / (mu_spread + sigma_spread)
    kappa_base = 1.2
    kappa = kappa_base * (0.5 + 1.5 * uncertainty_ratio)

    ucb = f_best - mu + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)

    # ---------- Combine EI and UCB with adaptive weights ----------
    def _normalise(arr):
        m = np.max(arr)
        if m > 1e-18:
            return arr / m
        return arr

    ei_n = _normalise(ei_blended)
    ucb_n = _normalise(ucb)

    # Weights: as the budget is consumed, shift from exploration to exploitation
    w_ei = 0.40 + 0.30 * norm_cost     # 0.40 → 0.70
    w_ucb = 1.0 - w_ei                 # 0.60 → 0.30

    scores = w_ei * ei_n + w_ucb * ucb_n

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

    return scores
# EVOLVE-BLOCK-END
