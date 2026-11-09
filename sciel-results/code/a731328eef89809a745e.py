"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Robust Expected Improvement (EI) with safe numerical handling.
    # Higher score => evaluate next.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe

    # Use stable computations for norm.cdf and norm.pdf
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z

    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Optionally add a small exploration term to break ties and encourage diversity
    # This is a mild UCB-like bonus: sigma * sqrt(log(1 + something))
    # But since we don't know the step count, use a very small constant multiplier
    # to avoid over-exploration.
    # Actually, pure EI is standard and effective. Let's stick with it but
    # ensure numerical stability.

    return ei
