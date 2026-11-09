"""Acquisition for Bayesian optimization of Weierstrass 1d; EVOLVE-BLOCK rewritten."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t


def acquisition(mu, sigma, f_best):
    # Robust standard deviation: floor to avoid division blowups / NaNs.
    s = np.asarray(sigma, dtype=float)
    s = np.where(s > 1e-8, s, 1e-8)

    # --- Component 1: Student-t Expected Improvement (heavy tails) ---
    # Improvement = f_best - mu (we minimize). Under a t posterior with
    # degrees of freedom nu, E[Im+] = (f_best - mu) * F_t(z) + s * nu *
    # (1 + z^2/nu)^(-(nu+1)/2) / nu, which is the t-analog of the Gaussian EI.
    nu = 3.0
    z = (f_best - mu) / s
    # t CDF and PDF
    F = student_t.cdf(z, df=nu)
    # t PDF: nu*(1+z^2/nu)^(-(nu+1)/2) / (nu*sqrt(pi/(nu-1)) * B(nu/2, 1/2))
    # Use scipy pdf for correctness.
    pdf = student_t.pdf(z, df=nu)
    ei_t = (f_best - mu) * F + s * pdf

    # --- Component 2: Adaptive UCB exploration bonus ---
    # Exploration weight decays as f_best -> 0 (known minimum). When f_best
    # is large (early), exploration dominates; when small (late), exploitation
    # of the best mean dominates. beta is a modest constant.
    beta = 0.4
    # Normalize exploration by a running scale so the bonus is meaningful in
    # objective units. Use f_best itself as a quality gate: when f_best is
    # near 0, the bonus vanishes.
    explore_gate = min(max(f_best, 0.0), 1.0)  # 1 early, ->0 late
    ucb = mu + beta * explore_gate * s

    # --- Combine: heavy-tailed EI drives the score; add a small UCB term ---
    # Weight the UCB term down so EI remains the primary signal but UCB
    # provides extra push to uncertain distant regions early on.
    score = ei_t + 0.3 * explore_gate * (ucb - mu)

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
