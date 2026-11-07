"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """Probability of Improvement with a mild exploration bonus.

    PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    We add a small exploration term proportional to sigma to encourage
    visiting under-explored regions, which helps with the low evaluation
    budget (25 steps) on the 2D Ackley function.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe

    # Core PoI score: probability that this point beats f_best
    poi = norm.cdf(z)

    # Mild exploration bonus: rewards high uncertainty regions.
    # Scaled by 0.01 so it doesn't dominate the PoI signal but nudges
    # the search toward unexplored areas.
    exploration_bonus = 0.01 * sigma_safe

    scores = poi + exploration_bonus

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
