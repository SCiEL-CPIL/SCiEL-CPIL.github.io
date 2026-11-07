# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Expected Improvement (EI) as the primary exploitation signal.
    When f_best is very small (late stage), adds a small UCB-like exploration
    term to escape local plateaus, scaled to be negligible relative to EI.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Adaptive exploration bonus that only kicks in when f_best is very small
    # This helps escape flat regions near the optimum
    # The bonus is scaled to be small relative to EI when f_best is large
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)

    # Exploration term: small UCB-like bonus that is more relevant when
    # f_best is small (late stage) and EI values are tiny
    # Use a smooth transition: exploration weight increases as f_best -> 0
    # But keep it very small to not disrupt EI's exploitation signal
    exploration_weight = 0.1 * (1.0 - ratio)  # 0.1 when f_best=0, 0 when f_best=25

    # UCB-like term for minimization: -mu + beta * sigma
    # But we only use the sigma part as an exploration bonus
    exploration_bonus = exploration_weight * sigma

    score = ei + exploration_bonus

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)

    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return EI only (fastest path)
        return ei

    return score
# EVOLVE-BLOCK-END
