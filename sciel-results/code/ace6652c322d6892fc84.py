import numpy as np
import time

# Deadline guard: evaluator kills after 330s wall-clock.
# Track time from module import; if we exceed the safety margin, return a
# safe fallback so we never get killed mid-computation.
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 5s safety margin before the 330s kill


def _time_exceeded():
    return (time.time() - _START_TIME) > _TIME_LIMIT


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.

    Structural approach: Pareto-frontier filtering + pure unnormalized UCB.
    1. Mask out candidates whose predicted mean mu is worse than f_best by
       more than a small tolerance (no plausible improvement there).
    2. On the remaining "plausible" subset, score with a pure, unnormalized
       UCB = -mu + beta * sigma to aggressively exploit high-uncertainty areas.
    3. Masked candidates get -inf so they are never selected.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Deadline guard: if we're about to be killed, return a safe fallback
    # (the lowest predicted mean) rather than risking a timeout.
    if _time_exceeded():
        fallback = -mu
        fallback = np.where(np.isfinite(fallback), fallback, -np.inf)
        return fallback

    # Floor sigma to avoid numerical issues.
    sigma_safe = np.maximum(sigma, 1e-12)

    # --- Step 1: Pareto-frontier filtering ---
    # A candidate is "plausible" if its predicted mean is not worse than
    # f_best by more than a tolerance. For minimization, mu should be
    # <= f_best + tol to be worth considering.
    #
    # The tolerance scales with the current best value. Early on (f_best large)
    # we are more permissive; late on (f_best near 0) we are stricter.
    # Weierstrass on [-0.5, 0.5] has values in roughly [0, ~0.5].
    f_scale = 0.5
    f_norm = min(max(f_best, 0.0), f_scale) / f_scale  # in [0, 1]

    # Tolerance: allow candidates whose mu is up to tol above f_best.
    # Larger tol early (explore more), smaller tol late (exploit more).
    tol = 0.05 + 0.45 * f_norm  # ranges from 0.05 (late) to 0.50 (early)

    plausible = mu <= (f_best + tol)

    # --- Step 2: Pure unnormalized UCB on the plausible subset ---
    # beta controls exploration. Higher when f_best is large (far from optimum),
    # lower when f_best is small (close to optimum).
    beta = 1.0 + 2.0 * f_norm  # ranges from 1.0 (late) to 3.0 (early)

    UCB = -mu + beta * sigma_safe

    # --- Step 3: Apply the mask ---
    # Masked candidates get -inf; plausible candidates get their UCB score.
    score = np.where(plausible, UCB, -np.inf)

    # If the mask is too aggressive and removes everything, fall back to
    # the least-bad candidate so we always return something finite-ish.
    if not np.any(plausible):
        # Relax the mask: allow the top 10% of candidates by mu.
        if mu.size > 0:
            threshold = np.percentile(mu, 10)
            plausible = mu <= threshold
            score = np.where(plausible, UCB, -np.inf)
        else:
            score = UCB

    # Ensure finite scores (replace any remaining NaN/inf with -inf).
    score = np.where(np.isfinite(score), score, -np.inf)

    return score
