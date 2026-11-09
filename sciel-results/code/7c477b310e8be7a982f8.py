"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known global minimum of Styblinski-Tang 1D
ST_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition using Monte Carlo estimation.
    Approximates the expected improvement in the global minimum by
    sampling from the posterior predictive at top-k candidates.
    
    mu, sigma: arrays of shape (2000,) in objective units.
    f_best: lowest observed value so far.
    
    Returns: array of 2000 finite scores (higher = evaluate next).
    """
    t0 = time.time()
    n = len(mu)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Select top-k candidates by a quick heuristic: low mu + high sigma
    # Use a simple score to pick candidates for MC sampling
    quick_score = -mu + 0.5 * sigma_safe
    k = min(20, n)
    top_k_idx = np.argpartition(quick_score, -k)[-k:]
    
    # Number of MC samples - keep small for speed
    n_samples = 50
    
    # Precompute the top-k mu and sigma
    top_mu = mu[top_k_idx]
    top_sigma = sigma_safe[top_k_idx]
    
    # Draw MC samples from the posterior predictive for top-k candidates
    # Shape: (n_samples, k)
    samples = top_mu[None, :] + top_sigma[None, :] * np.random.randn(n_samples, k)
    
    # For each sample, the new best would be min(current f_best, min over sampled values)
    # The improvement for candidate i is: max(0, f_best - min(samples[:, i]))
    # But we need to account for the fact that other candidates might also improve
    
    # KG approximation: for each candidate i, estimate E[max(0, f_best - min_i(f_i))]
    # where the min is over all candidates, but we approximate with top-k
    
    # For each candidate i in top-k, compute the expected improvement
    # kg_score[i] = E[max(0, f_best - min_j(samples[:, j]))]
    # This is the same for all i in a given sample, so we compute it once per sample
    
    # Actually, a better KG approximation:
    # For candidate i, the KG value is E[f_best - min(f_best, f_i, min_{j≠i} f_j)]
    # = E[max(0, f_best - f_i, f_best - min_{j≠i} f_j)]
    
    # Simplified: for each candidate i, compute
    # E[max(0, f_best - min(f_i, min_{j≠i} f_j))]
    # where the expectation is over the joint posterior
    
    # Let's compute for each candidate i in top_k:
    # kg_i = mean over samples of max(0, f_best - min(samples[:, i], min of other top-k in that sample))
    
    # To be efficient, for each sample row, compute the min over all top-k
    # Then for candidate i, the improvement is max(0, f_best - min(samples[:, i], min of others))
    
    # But this is complex. Let's use a simpler approximation:
    # KG_i ≈ E[max(0, f_best - f_i)] - E[max(0, f_best - min_{j≠i} f_j)]
    # This is the marginal gain of knowing f_i vs not knowing it.
    
    # Even simpler: use the standard EI as a proxy, but weight by uncertainty
    # Or: compute the probability that candidate i is the best among top-k
    
    # Let's try: for each candidate i, compute
    # P(i is best among top-k) * EI_i + (1 - P(i is best)) * small_value
    
    # Compute probability that each top-k candidate is the minimum
    # P(i is min) = mean over samples of [samples[:, i] == min(samples, axis=1)]
    
    min_per_sample = np.min(samples, axis=1)  # shape (n_samples,)
    is_min = (samples == min_per_sample[:, None])  # shape (n_samples, k)
    prob_min = is_min.mean(axis=0)  # shape (k,)
    
    # Compute EI for top-k candidates
    z = (f_best - top_mu) / top_sigma
    z_clipped = np.clip(z, -10, 10)
    ei_top = (f_best - top_mu) * norm.cdf(z_clipped) + top_sigma * norm.pdf(z_clipped)
    
    # KG score for top-k: weight EI by probability of being the best
    kg_top = prob_min * ei_top
    
    # For non-top-k candidates, use a smaller EI-based score
    z_all = (f_best - mu) / sigma_safe
    z_all_clipped = np.clip(z_all, -10, 10)
    ei_all = (f_best - mu) * norm.cdf(z_all_clipped) + sigma_safe * norm.pdf(z_all_clipped)
    
    # Scale EI_all to be smaller than the max kg_top
    max_kg = np.max(kg_top) if len(kg_top) > 0 else 1.0
    if max_kg > 0:
        scale_factor = max_kg / (np.max(ei_all) + 1e-9)
        score_non_top = ei_all * scale_factor * 0.3  # dampen non-top-k
    else:
        score_non_top = ei_all * 0.01
    
    # Build the full score array
    score = np.zeros(n)
    score[top_k_idx] = kg_top
    score[np.setdiff1d(np.arange(n), top_k_idx)] = score_non_top[np.setdiff1d(np.arange(n), top_k_idx)]
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() - t0 > 2.5:
        return score
    
    return score
# EVOLVE-BLOCK-END
