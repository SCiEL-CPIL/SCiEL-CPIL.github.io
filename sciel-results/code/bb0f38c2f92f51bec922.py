"""Rewritten cost-aware acquisition for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock to stay safely under the 330 s evaluator kill.
_start_time = time.time()
TIME_LIMIT = 325.0  # small safety margin


def _deadline_exceeded():
    return (time.time() - _start_time) > TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Levy 2D on [-10, 10]^2.

    Structural change from the previous EI + kappa*sigma design:
      - Use a log-normal surrogate for the objective (more natural for
        positive, skewed functions like Levy).
      - Acquisition = expected log-improvement over the best log-objective,
        divided by a cost-aware penalty.
      - Add a UCB-style uncertainty bonus scaled by the GP sigma.
      - Cost penalty uses a soft-max (exp) to strongly prefer cheaper
        evaluations while remaining smooth.

    Returns: array of 2000 finite scores (higher = evaluate next).
    """
    # Fast path: if we are close to the deadline, return a cheap heuristic
    # so we never risk a timeout.
    if _deadline_exceeded():
        # Simple UCB/cost as a last-resort fallback.
        z = (f_best - mu) / (sigma + 1e-12)
        ucb = mu + 1.0 * sigma  # UCB-style (for minimization we want low mu, high sigma)
        # Convert to a "higher is better" score.
        score = -ucb / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)

    # --- Log-normal surrogate acquisition ---
    # The GP is fit to standardized targets, but mu and sigma are in objective
    # units per the task description.  We treat mu as the predicted objective
    # and sigma as the predicted standard deviation.

    # Clip mu to a small positive floor so log(mu) is well-defined.
    mu_safe = np.maximum(mu, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-6)

    # Log-objective: y = log(f)
    # If f ~ LogNormal(log_mu, sigma_f^2), then log(f) ~ Normal(log_mu, sigma_f^2)
    # We approximate log_mu = log(mu_safe) and sigma_log = sigma_safe / mu_safe
    # (delta-method approximation for small relative uncertainty).
    log_mu = np.log(mu_safe)
    sigma_log = sigma_safe / mu_safe

    # Best observed log-objective.
    log_f_best = np.log(max(f_best, 1e-6))

    # Expected log-improvement (minimization in log-space):
    #   E[max(log_f_best - Y, 0)] where Y ~ N(log_mu, sigma_log^2)
    z = (log_f_best - log_mu) / (sigma_log + 1e-12)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    e_log_improvement = (log_f_best - log_mu) * cdf + sigma_log * pdf

    # UCB-style uncertainty bonus: reward high sigma to explore.
    # Scale by the typical magnitude of log-objective to keep units consistent.
    ucb_bonus = 0.5 * sigma_log  # modest exploration

    # Combined acquisition in log-space.
    acq_log = e_log_improvement + ucb_bonus

    # Cost-aware penalty: prefer cheaper evaluations.
    # Use a soft exponential penalty so the score is smooth.
    # cost is in [1, 2] (1 + mean(x_scaled)).
    # We want to strongly discount expensive points.
    cost_penalty = np.exp(0.5 * (cost - 1.0))  # ranges from 1 to ~exp(0.5)≈1.65

    # Final score: higher is better.
    acq = acq_log / (cost_penalty + 1e-12)

    # Ensure all values are finite.
    acq = np.where(np.isfinite(acq), acq, 0.0)

    return acq
# EVOLVE-BLOCK-END
