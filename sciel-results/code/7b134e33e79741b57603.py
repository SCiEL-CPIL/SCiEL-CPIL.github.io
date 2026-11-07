"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s limit


def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe default if we're running out of time
    if _check_deadline():
        # Return cost-based scores (prefer low cost) as a safe fallback
        cost = np.asarray(cost, dtype=np.float64)
        return 1.0 / (cost + 1e-9)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)

    n = len(mu)

    # Compute Expected Shortfall (CVaR) - average of worst 10% of samples
    # For a Gaussian posterior, the worst 10% corresponds to the lower tail
    # CVaR_0.1 = E[f | f <= VaR_0.1]
    # For N(mu, sigma^2): VaR_0.1 = mu + sigma * z_0.1 where z_0.1 is the 10th percentile
    # CVaR_0.1 = mu + sigma * E[Z | Z <= z_0.1]
    # E[Z | Z <= z_0.1] = -phi(z_0.1) / 0.1 where phi is standard normal pdf
    z_cvar = norm.ppf(0.10)  # approximately -1.2816
    phi_z = norm.pdf(z_cvar)
    cvar_coeff = -phi_z / 0.10  # This is E[Z | Z <= z_0.1]

    # CVaR of the objective at each candidate
    cvar = mu + sigma * cvar_coeff

    # Also compute Expected Improvement for comparison
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)

    # Compute UCB for exploration
    # Use adaptive beta: higher uncertainty exploration early, lower later
    # Use f_best as a proxy for progress (lower f_best = more progress)
    # But we don't have iteration count, so use cost distribution
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range

    # Beta starts high for exploration, decreases for exploitation
    beta = 2.5 - 1.5 * cost_norm
    ucb = mu + beta * sigma

    # Bayesian Risk Minimization: use CVaR as the primary risk measure
    # Normalize CVaR to [0, 1] (lower is better, so invert)
    cvar_min = np.min(cvar)
    cvar_max = np.max(cvar)
    cvar_range = cvar_max - cvar_min + 1e-9
    cvar_norm = (cvar - cvar_min) / cvar_range  # 0 = best, 1 = worst
    cvar_score = 1.0 - cvar_norm  # Higher is better

    # Normalize EI
    ei_max = np.max(ei) + 1e-9
    ei_norm = ei / ei_max

    # Normalize UCB (lower mu + beta*sigma is better for minimization)
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range
    ucb_score = 1.0 - ucb_norm

    # Combine: primary is CVaR (risk-adjusted), secondary is EI and UCB
    # Weight CVaR heavily as it hedges against multi-modal traps
    cvar_weight = 0.5
    ei_weight = 0.3
    ucb_weight = 0.2

    combined = cvar_weight * cvar_score + ei_weight * ei_norm + ucb_weight * ucb_score

    # Cost-aware penalty: divide by cost to prefer low-cost evaluations
    # Use a softer penalty to allow some exploration of high-cost but promising points
    score = combined / (cost + 1e-9)

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
