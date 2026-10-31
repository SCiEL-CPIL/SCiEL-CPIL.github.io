"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import weibull_min


def _weibull_cdf(x, a=1.5, b=20.0):
    """CDF of Weibull distribution with shape a and scale b."""
    if x <= 0:
        return 0.0
    return 1.0 - np.exp(-(x / b) ** a)


def _expected_regret(remainder, a=1.5, b=20.0):
    """
    Estimate expected future regret for a given remainder.
    A remainder is 'good' if it can be filled by items from the Weibull distribution.
    We approximate this by the probability that a random item fits in the remainder,
    weighted by the remainder size.
    """
    if remainder <= 0:
        return 0.0
    # Probability that a single item fits in the remainder
    p_fit = _weibull_cdf(remainder, a, b)
    # Expected number of items that can fit (rough approximation)
    expected_items = p_fit * (remainder / b)
    # Regret is higher when we can't fill the bin well
    # Use a smooth penalty: regret increases with remainder but decreases with fill probability
    return remainder * (1.0 - p_fit)


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on expected future regret of the remaining capacity.
    Lower regret is better, so we return negative regret as the score.
    """
    remainders = bins - item
    valid = remainders >= 0
    
    # Calculate expected regret for each valid bin
    regrets = np.zeros(len(bins))
    if np.any(valid):
        valid_remainders = remainders[valid]
        # Use vectorized Weibull CDF
        p_fit = 1.0 - np.exp(-(valid_remainders / 20.0) ** 1.5)
        regrets[valid] = valid_remainders * (1.0 - p_fit)
    
    # Return negative regret (higher score = better)
    return -regrets
# EVOLVE-BLOCK-END
