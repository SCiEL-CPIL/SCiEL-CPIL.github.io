import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)  # cost is in [1, 2]

    # Check deadline
    if time.time() > deadline:
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -50, 50)
        phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
        cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
        ei = (f_best - mu) * cdf_z + sigma * phi_z
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)

    # Compute phi(z) and Phi(z)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))

    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement
    poi = cdf_z

    # Blend EI with PoI-based confidence factor
    # (1 + PoI) amplifies scores for points likely to improve
    blended_ei = ei * (1.0 + poi)

    # Check deadline
    if time.time() > deadline:
        scores = blended_ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Cost-aware scoring with adaptive exponent
    # The cost is 1 + mean(x_scaled), so it ranges from 1 to 2
    # We want to penalize higher cost evaluations more aggressively as we converge

    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min

    if mu_range > 1e-9:
        f_best_rel = (f_best - mu_min) / mu_range
        f_best_rel = np.clip(f_best_rel, 0, 1)
    else:
        f_best_rel = 0.5

    # Cost exponent: lower when exploring, higher when exploiting
    # Range: [0.8, 1.8]
    cost_exp = 0.8 + 1.0 * f_best_rel

    # Uncertainty-weighted adjustment
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)

    if sigma_max > 1e-9:
        sigma_rel = sigma / sigma_max
    else:
        sigma_rel = np.zeros_like(sigma)

    # Per-point cost exponent adjustment:
    # High uncertainty -> lower cost penalty (more exploration)
    # Low uncertainty -> higher cost penalty (more exploitation)
    cost_exp_adjusted = cost_exp * (0.7 + 0.3 * (1.0 - sigma_rel))

    # Base score: blended EI normalized by cost with adaptive exponent
    scores = blended_ei / (cost_safe ** cost_exp_adjusted)

    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)

    # Add a small exploration bonus for high-uncertainty points
    # This helps escape local optima in the 4D landscape
    exploration_bonus = 0.05 * sigma_rel * blended_ei

    scores += exploration_bonus

    # Also compute a UCB-based score and take the max
    # UCB = -mu + kappa * sigma
    # Adaptive kappa based on convergence

    improvement_spread = np.std(f_best - mu)
    if improvement_spread > 1e-9:
        convergence_ratio = sigma_mean / improvement_spread
    else:
        convergence_ratio = 1.0

    # kappa: [0.5, 3.0] - higher when less converged
    kappa = 0.5 + 2.5 * np.clip(1.0 - convergence_ratio, 0, 1)

    # UCB score (for minimization, higher is better):
    ucb = -mu + kappa * sigma
    ucb = np.maximum(ucb, 0)

    # Normalize UCB to be comparable to EI scale
    max_blended_ei = np.max(blended_ei)
    max_ucb = np.max(ucb)
    if max_ucb > 1e-9 and max_blended_ei > 1e-9:
        ucb_scale = max_blended_ei / max_ucb
    else:
        ucb_scale = 1.0

    ucb_score = ucb * ucb_scale / (cost_safe ** cost_exp_adjusted)

    # Final score: max of EI-based and UCB-based
    scores = np.maximum(scores, ucb_score)

    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Final deadline check
    if time.time() > deadline:
        return scores

    return scores
