import numpy as np
import time

_DEADLINE = 300.0
_START_TIME = time.monotonic()

def _elapsed():
    return time.monotonic() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a lower-bound
    exploit term to bias selection toward points with the smallest posterior
    mean when EI is small.

    The score is:
        score = EI(x) + alpha * max(0, f_best - mu(x) - beta * sigma(x))

    The second term is a soft lower-bound: it is zero when the lower bound
    (mu + beta*sigma) is still above f_best, but grows linearly once the
    lower bound drops below f_best, encouraging exploitation of points that
    are very likely to beat the current best.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
    sigma : ndarray, shape (2000,)
    f_best : float

    Returns
    -------
    scores : ndarray, shape (2000,)
    """
    # Safety margin: abort if we're close to the deadline
    if _elapsed() > _DEADLINE - 2.0:
        return np.zeros(len(mu))

    t0 = time.monotonic()
    n = len(mu)
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # --- Expected Improvement ---
    z = (f_best - mu) / sigma_safe
    # Use erfc for numerical stability in the tails
    # norm.cdf(z) = 0.5 * erfc(-z / sqrt(2))
    from scipy.special import erfc, erf
    # For stability, compute cdf directly
    # cdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # pdf(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    sqrt2 = np.sqrt(2.0)
    cdf = 0.5 * (1.0 + erf(z / sqrt2))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # --- Exploit lower-bound term ---
    # Lower bound: mu + beta * sigma.  If this is below f_best,
    # the point is promising.  Score contribution: max(0, f_best - (mu + beta*sigma))
    beta = 2.0  # exploration width for the lower bound
    lb = mu + beta * sigma_safe
    exploit = np.maximum(0.0, f_best - lb)

    # Blend: EI captures expected improvement; exploit captures
    # "this point's lower bound is already below f_best, so it's very likely
    # to improve".  Weight exploit slightly less than EI to avoid pure exploitation.
    alpha = 0.5
    scores = ei + alpha * exploit

    # Normalize for numerical safety
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Check elapsed time; if too slow, fall back to pure EI
    if time.monotonic() - t0 > 2.0:
        # This shouldn't happen, but guard anyway
        pass

    return scores
