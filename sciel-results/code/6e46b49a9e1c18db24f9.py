"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses a Knowledge Gradient (KG) approximation that estimates the expected
    reduction in the global minimum by combining expected improvement over
    f_best with a soft-min surrogate over the predictive distribution.
    """
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    n = len(mu)

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)

    # Progress estimate: how close f_best is to the known minimum (0)
    # Larger f_best means more room for improvement (exploration phase)
    # Smaller f_best means we're exploiting near the minimum
    progress = 1.0 - f_best / (f_best + 1.0)
    progress = np.clip(progress, 0.0, 1.0)

    # Knowledge Gradient approximation:
    # For each candidate, estimate E[max(0, f_best - Y)] where Y ~ N(mu, sigma^2)
    # This is the expected improvement (EI) for each point.
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF.

    from scipy.stats import norm

    # Compute z-scores
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(sigma > 1e-12, (f_best - mu) / sigma, np.inf)

    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)

    # Expected improvement for each point
    with np.errstate(invalid='ignore'):
        ei = np.where(sigma > 1e-12,
                      (f_best - mu) * Phi_z + sigma * phi_z,
                      np.maximum(0.0, f_best - mu))

    ei = np.maximum(0.0, ei)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    # Soft-min surrogate: approximate the expected minimum over the 2000 samples.
    # Use log-sum-exp trick for numerical stability.
    # The soft-min is -logsumexp(-mu / tau) * tau for temperature tau.
    # As tau -> 0, this approaches min(mu).
    # We use a temperature that adapts: higher temperature early (more exploration),
    # lower temperature late (sharper exploitation).
    tau = 0.5 * (1.0 - 0.7 * progress) + 0.1
    # tau ranges from ~0.6 early to ~0.1 late

    # Soft-min of mu values (lower is better, so we want the minimum)
    # soft_min = -tau * logsumexp(-mu / tau)
    scaled_mu = -mu / tau
    # Subtract max for numerical stability
    max_scaled = np.max(scaled_mu)
    logsumexp_val = max_scaled + np.log(np.sum(np.exp(scaled_mu - max_scaled)))
    soft_min_val = -tau * logsumexp_val

    # The KG score should reward points that are likely to reduce the global minimum.
    # A point reduces the global minimum if it beats the current soft-min estimate.
    # We combine:
    # 1. Expected improvement over f_best (direct exploitation signal)
    # 2. Probability that this point is the new minimum (KG-like signal)
    # 3. Uncertainty bonus that decays as we progress

    # Adaptive exploration coefficient for the uncertainty term
    kappa = 2.0 - 1.5 * progress  # ranges from 2.0 to 0.5

    # Primary score: EI (expected improvement over f_best)
    score = ei

    # Secondary: reward points with low predicted mean (exploitation)
    # Weighted by how much they beat the soft-min
    below_soft_min = np.maximum(0.0, soft_min_val - mu)
    score = score + 0.3 * below_soft_min

    # Tertiary: uncertainty bonus (exploration), decays over time
    score = score + kappa * sigma * 0.3

    # Normalize to ensure finite and reasonable scale
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
