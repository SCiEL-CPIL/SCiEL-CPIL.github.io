"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_KG_CACHE = {}

def acquisition(mu, sigma, f_best):
    """Knowledge Gradient acquisition for Bayesian optimization."""
    n = len(mu)
    # Safety: ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    # Replace non-finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Fast screening: compute EI for all candidates
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Select top-k candidates for KG evaluation
    k = min(100, n)
    top_idx = np.argpartition(-ei, k - 1)[:k]
    top_mu = mu[top_idx]
    top_sigma = sigma[top_idx]

    # Monte Carlo samples for KG
    n_samples = 50
    rng = np.random.default_rng(42)

    # For each top candidate, estimate KG = E[f_best - min(f_best, f(x_i))]
    # where f(x_i) ~ N(mu_i, sigma_i)
    # KG_i = E[max(f_best - f(x_i), 0)] is just EI_i
    # But true KG considers that evaluating x_i updates the posterior,
    # potentially improving the best in the whole set.
    # Approximation: KG_i ≈ E[max(f_best - min(f(x_i), f_best), 0)]
    # plus a bonus for reducing uncertainty in the global minimum.

    # Better approximation: sample from each candidate's posterior and
    # estimate the expected improvement in the global minimum.
    # For candidate i, if we evaluate it, the new best is min(f_best, f(x_i)).
    # The expected reduction in regret is E[max(f_best - min(f_best, f(x_i)), 0)]
    # which equals EI_i. But KG also accounts for information value.

    # Practical KG approximation:
    # KG_i = E[ f_best - min(f_best, f(x_i)) ] + small info bonus
    # = EI_i + info_bonus_i

    # A more faithful KG: consider that after evaluating x_i, we update beliefs
    # about all candidates. The global minimum's expected value decreases.
    # Approximate by: for each MC sample, compute the expected minimum over
    # all candidates given that x_i was observed.

    # Simplified but effective KG:
    # 1. For each top candidate i, draw samples from N(mu_i, sigma_i)
    # 2. For each sample, the new best is min(f_best, sample)
    # 3. The expected improvement in best is E[f_best - min(f_best, sample)] = EI_i
    # 4. Add an information bonus: candidates with high uncertainty in promising
    #    regions are valuable because they reduce the variance of the global min.

    # Let's use a more principled approach:
    # KG_i = E[max(0, f_best - f(x_i))] - E[max(0, f_best - f(x_i)) | not evaluating i]
    # This is complex. Use the common approximation:
    # KG_i ≈ EI_i * (1 + c * sigma_i / (f_best - mu_i + eps)) for promising candidates

    # Actually, let me use the standard KG Monte Carlo:
    # For candidate i, sample S ~ N(mu_i, sigma_i)
    # After observing S, the posterior mean of the global minimum decreases.
    # Approximate: the new expected minimum is min over all j of E[f(x_j) | S]
    # For j != i, E[f(x_j) | S] ≈ mu_j (weak dependence in 1D with short length scale)
    # For j = i, E[f(x_i) | S] = S
    # So new expected min ≈ min(f_best, S, min_{j≠i} mu_j)
    # KG_i = E[f_best - min(f_best, S, min_{j≠i} mu_j)]

    # Compute min_mu_others for each top candidate
    # min over all candidates except top_idx[i]
    # Efficient: sort mu, find min excluding each top candidate

    mu_sorted = np.sort(mu)
    min_mu_all = mu_sorted[0]
    second_min_mu = mu_sorted[1] if n > 1 else mu_sorted[0]

    kg_values = np.zeros(k)
    # Draw samples for all top candidates at once
    samples = rng.normal(top_mu[:, None], top_sigma[:, None], size=(k, n_samples))

    for i in range(k):
        # min over others: if mu[i] is the global min, use second min, else use global min
        if top_mu[i] == min_mu_all:
            min_others = second_min_mu
        else:
            min_others = min_mu_all

        # New best after evaluating candidate i with sample s:
        # min(f_best, s, min_others)
        new_best = np.minimum(f_best, np.minimum(samples[i], min_others))
        kg_values[i] = np.mean(f_best - new_best)

    # Map back to full array
    scores = np.zeros(n)
    scores[top_idx] = kg_values

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
