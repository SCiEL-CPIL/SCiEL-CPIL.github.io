import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive hybrid acquisition: blends Expected Improvement with an
    exploration bonus whose magnitude adapts to how many candidates
    currently look promising (mu < f_best).
    
    When few candidates look promising (early / under-explored), the
    exploration bonus is larger, encouraging broader search. As more
    candidates fall below f_best, the bonus shrinks and the score
    converges to pure EI (exploitation).
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Fraction of candidates that look better than f_best
    promising_frac = np.mean(mu < f_best)

    # Adapt beta: high when few promising points (explore), low when many (exploit)
    # promising_frac in [0,1]; map to beta in [0.1, 0.5]
    beta = 0.5 - 0.4 * promising_frac

    # Hybrid score: EI + beta * sigma (exploration bonus)
    scores = ei + beta * sigma_safe

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
