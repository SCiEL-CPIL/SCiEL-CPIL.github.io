# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    cost = np.maximum(cost, 1e-9)

    # Calculate z.
    z = (f_best - mu) / (sigma + 1e-9)

    # Calculate Expected Improvement.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Calculate Probability of Improvement (PI).
    pi = norm.cdf(z)

    # Expected Minimum (EM) improvement:
    # E[max(0, f_best - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is the same as EI but we use a slightly different formulation
    # that emphasizes the expected value of the minimum.
    # EM = -sigma * phi(z) + (f_best - mu) * Phi(z) - sigma * phi(z)
    # Actually, let's use a different approach.

    # Let's try a hybrid: combine EI with a lower confidence bound (LCB) term.
    # LCB: mu - kappa * sigma (lower is better for minimization)
    # We want to maximize the expected reduction, so we use -LCB = -mu + kappa*sigma

    # The key insight: use a weighted combination that adapts based on
    # the relationship between EI and uncertainty.
    # When sigma is large relative to (f_best - mu), exploration is valuable.
    # When (f_best - mu) is large, exploitation is valuable.

    # Define a normalized uncertainty ratio.
    # When z is small (high uncertainty relative to improvement), favor exploration.
    # When z is large (high confidence in improvement), favor exploitation.

    # Use a smooth transition: the weight on the exploration term
    # decreases as z increases.
    # w_explore = 1 / (1 + z^2) for z >= 0, and 1 for z < 0
    # But this can be noisy, so let's use a softer version.

    # Actually, let's try a different structural approach:
    # Use the "Expected Improvement with Adaptive Cost" (EIA) which
    # multiplies EI by a factor that accounts for the cost structure.

    # The cost is 1 + mean(x_scaled), where x_scaled = (x+5)/10
    # So cost ranges from 1 to 2.
    # We want to prefer cheaper points when the improvement is similar.

    # Let's try: score = EI / cost^alpha, where alpha > 1
    # This penalizes expensive points more heavily.
    alpha = 1.5
    score = ei / (cost ** alpha)

    # Add a small exploration bonus scaled by sigma
    # The bonus should be larger when uncertainty is high
    exploration_bonus = 0.03 * sigma / (cost ** 0.5)

    score = score + exploration_bonus

    # Ensure the scores are finite.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
