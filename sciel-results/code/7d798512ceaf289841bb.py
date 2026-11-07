"""Cost-aware acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_MARGIN_S = 5.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _deadline_reached():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN_S)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe finite score that picks the lowest-cost candidate
    if _deadline_reached():
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e30, dtype=np.float64)
        scores[idx] = 0.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Cost-aware Expected Hypar Improvement (EHI)
    # EHI = E[tanh((f_best - Y) / (f_best + cost*epsilon))]
    # where Y ~ N(mu, sigma^2)
    # This formulation naturally balances improvement and cost.

    # The hyperparameter for EHI: gamma controls the scale of the tanh argument
    # A typical value is related to the problem scale
    # For Hartmann 6d, the minimum is about -3.32, so f_best is in that range
    # We use a normalization that makes the EHI meaningful

    # EHI formula:
    # E[tanh(a * (f_best - Y) / b)] where a and b are hyperparameters
    # With Y ~ N(mu, sigma^2):
    # E[tanh((f_best - Y)/c)] = integral of tanh((f_best - y)/c) * N(y; mu, sigma^2) dy

    # For computational tractability, we use an approximation:
    # When sigma is small relative to c, tanh((f_best - mu)/c) is a good approximation
    # When sigma is large, the expectation approaches 0 (symmetric cancellation)

    # A better approach: use the fact that for normal Y,
    # E[tanh((f_best - Y)/c)] can be approximated using the error function

    # Actually, let's use a simpler but effective formulation:
    # EHI ≈ (f_best - mu) * exp(-sigma^2 / (2*c^2)) / c
    # This is a Gaussian-smoothed version of the improvement

    # The cost-aware version divides by a function of cost
    # EHI_cost = EHI / (1 + lambda * (cost - 1))
    # where lambda controls cost sensitivity

    c = 0.5  # scale parameter for the tanh-like smoothing
    lambda_cost = 2.0  # cost sensitivity

    # Compute the smoothed improvement
    # For small sigma: ~ (f_best - mu) / c
    # For large sigma: ~ (f_best - mu) * exp(-sigma^2/(2c^2)) / c
    improvement = f_best - mu
    gaussian_damping = np.exp(-sigma**2 / (2 * c**2))
    ehi = improvement * gaussian_damping / c

    # Cost penalty: higher cost reduces the score
    # cost is in [1, 2] (since 1 + mean(x) where x in [0,1]^6)
    # Normalize cost to [0, 1] range: (cost - 1) is in [0, 1]
    cost_normalized = cost - 1.0  # now in [0, 1]
    cost_penalty = 1.0 + lambda_cost * cost_normalized

    # Final score: EHI divided by cost penalty
    score = ehi / cost_penalty

    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e30)

    # Clip to avoid extreme values
    score = np.clip(score, -1e10, 1e10)

    return score
# EVOLVE-BLOCK-END
