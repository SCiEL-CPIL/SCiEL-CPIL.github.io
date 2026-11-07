"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Structure-aware cost-weighted hybrid acquisition: blends EI and UCB with
    adaptive cost penalty that strengthens early (exploration) and weakens as
    the GP converges (exploitation).
    """
    start_time = time.time()
    deadline = 300.0

    sigma = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-8)

    # --- Expected Improvement (minimization) ---
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # --- UCB (minimization: we want low mu, high sigma) ---
    # beta controls exploration; adapt based on GP convergence
    # Estimate convergence: if sigma is small relative to the scale of the problem,
    # the GP has converged -> reduce exploration
    sigma_scale = np.median(sigma)
    # f_best for ST2D is around -78; use |f_best| as a rough scale reference
    # but better: use the range of mu as a proxy for problem scale
    mu_range = np.ptp(mu)
    if mu_range < 1e-6:
        mu_range = max(abs(f_best), 1.0)
    # Convergence indicator: 0 = not converged, 1 = converged
    # If median sigma is small relative to mu_range, GP is confident
    conv = 1.0 - np.clip(sigma_scale / (mu_range + 1e-8), 0.0, 1.0)
    # When converged (conv~1), use less exploration; when not (conv~0), explore more
    beta = 0.5 + 1.5 * (1.0 - conv)  # beta in [0.5, 2.0]
    ucb = -mu + beta * sigma  # for minimization, lower mu and higher sigma are good

    # --- Adaptive cost weighting ---
    # Total cost is embedded in the cost array; use mean cost as a proxy for budget used
    mean_cost = np.mean(cost)
    # Budget is 30, initial 5 points cost ~1.5 each = ~7.5, so remaining ~22.5
    # As mean_cost increases (we're later in the budget), weight cost more heavily
    # cost_weight: 1.0 early (cost matters a lot), decreasing to ~0.3 late
    # Actually, we want to penalize cost more when we have less budget left
    # Let's use: cost_penalty = 1 / (cost^alpha) where alpha adapts
    # Early: alpha small (cost matters less, explore)
    # Late: alpha larger (cost matters more, exploit cheap points)
    budget_used_frac = np.clip((mean_cost - 1.0) / 29.0, 0.0, 1.0)
    alpha = 0.3 + 0.7 * budget_used_frac  # alpha in [0.3, 1.0]
    cost_factor = 1.0 / np.power(cost_safe, alpha)

    # --- Hybrid score ---
    # Blend EI and UCB: early on, UCB helps explore; later, EI exploits
    # weight_ei increases with convergence
    w_ei = 0.3 + 0.7 * conv
    w_ucb = 1.0 - w_ei

    # Normalize EI and UCB to comparable scales
    ei_scale = np.std(ei) + 1e-8
    ucb_scale = np.std(ucb) + 1e-8
    ei_norm = ei / ei_scale
    ucb_norm = ucb / ucb_scale

    hybrid = w_ei * ei_norm + w_ucb * ucb_norm

    # Apply adaptive cost weighting
    score = hybrid * cost_factor

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
