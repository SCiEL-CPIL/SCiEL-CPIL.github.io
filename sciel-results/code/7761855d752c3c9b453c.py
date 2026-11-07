import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) for minimization with a small UCB blend
    to retain exploration in low-uncertainty regions.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill

    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-9)

    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    #
    # Let Z = (f(x) - f_best) / sigma(x) ~ N((mu - f_best)/sigma, 1)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    #     = (f_best - mu) * Phi(d) + sigma * phi(d)
    # where d = (f_best - mu) / sigma

    d = (f_best - mu) / sigma

    # Phi(d) = CDF, phi(d) = PDF
    Phi_d = norm.cdf(d)
    phi_d = norm.pdf(d)

    # EI = (f_best - mu) * Phi_d + sigma * phi_d
    # When d is very negative (mu >> f_best), EI ≈ 0 (no improvement expected)
    # When d is very positive (mu << f_best), EI ≈ f_best - mu (high improvement)
    EI = (f_best - mu) * Phi_d + sigma * phi_d

    # Add a small UCB component for exploration
    # This ensures we don't completely ignore regions with high uncertainty
    # even when EI is zero (because mu >> f_best)
    # Scale the UCB term relative to the problem scale
    ucb_component = 0.3 * sigma

    # Combined score: EI dominates, UCB provides exploration floor
    scores = EI + ucb_component

    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Check if we're running out of time
    if time.time() > deadline:
        # Fallback: pure EI without UCB blend
        fallback_scores = EI
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return fallback_scores

    return scores
