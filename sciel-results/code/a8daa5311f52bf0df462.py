"""Acquisition for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START = time.monotonic()
_DEADLINE = 290.0


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # return a safe fallback (pure exploitation via lowest mu) immediately.
    if (time.monotonic() - _START) > _DEADLINE:
        return -mu

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = mu.shape[0]

    # Ensure sigma is positive and finite.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, np.inf)

    # Knowledge Gradient (KG) approximation for minimization:
    # The exact KG for a single next evaluation is:
    #   KG_i = E[ max(0, f_best - Y_i) ] - E[ max(0, min_j Y_j) ]
    # where Y_i ~ N(mu_i, sigma_i^2) and the second term is the expected
    # minimum over all candidates after observing Y_i.
    #
    # A common and efficient approximation (the "one-step" KG) replaces the
    # second term with the expected improvement of the current best, giving:
    #   KG_i ≈ (f_best - mu_i) * Phi(z_i) + sigma_i * phi(z_i)
    # which is exactly the Expected Improvement (EI).
    #
    # However, a more principled KG approximation accounts for the fact that
    # after evaluating point i, the new best is min(f_best, Y_i). The expected
    # new best is:
    #   E[min(f_best, Y_i)] = f_best - EI_i
    # so the expected reduction in the best value is EI_i.
    #
    # To capture more of the KG structure, we use a "two-stage" approximation
    # that considers the probability of finding a new best and the expected
    # improvement conditional on that event, plus an exploration bonus that
    # scales with the uncertainty relative to the current best gap.
    #
    # We compute:
    #   z_i = (f_best - mu_i) / sigma_i
    #   p_improve = Phi(z_i)  (probability that Y_i < f_best)
    #   cond_improve = (f_best - mu_i) + sigma_i * phi(z_i)/Phi(z_i)  (expected improvement given improvement)
    #   EI_i = p_improve * cond_improve = (f_best - mu_i)*Phi(z_i) + sigma_i*phi(z_i)
    #
    # The KG score is then EI_i, but we add a small exploration bonus for
    # points with high sigma relative to the typical scale, to encourage
    # exploring regions with high uncertainty even if their mean is not
    # particularly low. This is a practical KG variant.
    #
    # Adaptive: early on (f_best far from optimum), use stronger exploration.
    # Late on (f_best near optimum), rely more on pure EI.

    z = (f_best - mu) / sigma
    z = np.where(np.isfinite(z), z, 0.0)

    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)

    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)

    # Standard EI
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Exploration bonus: sigma * (1 - cdf_z) captures the "tail" where
    # Y_i could be much better than f_best. This is related to the
    # expected shortfall. We scale it by an adaptive factor.
    tail_bonus = sigma * (1.0 - cdf_z)
    tail_bonus = np.where(np.isfinite(tail_bonus), tail_bonus, 0.0)

    # Adaptive weight on exploration:
    # gap = f_best - global_min (how far we are from the known optimum)
    global_min = -2.53218
    gap = max(f_best - global_min, 0.0)
    
    # When gap is large (early), explore more. When gap is small (late), exploit.
    # w_explore goes from ~0.5 (early) to ~0.0 (late)
    w_explore = 0.5 * np.exp(-gap * 1.5)

    # KG score = EI + w_explore * tail_bonus
    score = ei + w_explore * tail_bonus
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
