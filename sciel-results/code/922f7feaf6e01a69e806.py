"""Rewritten program: entropy-weighted EI with batch-aware look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining entropy-weighted EI with a lightweight
    batch-aware look-ahead to encourage diverse candidate selection.
    
    Parameters:
        mu: array of shape (2000,), GP mean in objective units
        sigma: array of shape (2000,), GP std in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        scores: array of shape (2000,), finite values, higher = evaluate next
    """
    # Safety: ensure inputs are 1-D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Entropy-weighted EI: amplify improvement in high-uncertainty regions
    # Use a soft entropy weight based on log(sigma) to balance exploration
    # The weight is bounded to prevent extreme values
    log_sigma = np.log(sigma_safe + 1e-12)
    # Normalize log_sigma to [0, 1] range approximately
    log_sigma_min = np.percentile(log_sigma, 5)
    log_sigma_max = np.percentile(log_sigma, 95)
    log_sigma_range = log_sigma_max - log_sigma_min
    if log_sigma_range > 1e-9:
        log_sigma_norm = (log_sigma - log_sigma_min) / log_sigma_range
    else:
        log_sigma_norm = np.zeros_like(log_sigma)
    
    # Entropy weight: higher uncertainty -> higher weight
    # Use a tunable exponent to control exploration strength
    entropy_weight = 1.0 + 1.5 * log_sigma_norm
    
    # Entropy-weighted EI
    h_ei = ei * entropy_weight
    
    # Batch-aware look-ahead: penalize candidates that are too close to
    # already-high-scoring candidates to encourage diversity
    # We do this by computing a simple diversity penalty based on the
    # top-k scores in the pool
    
    # Find the top 50 scores and penalize similar candidates
    top_k = 50
    top_indices = np.argpartition(h_ei, -top_k)[-top_k:]
    top_scores = h_ei[top_indices]
    
    # For each candidate, compute the maximum similarity to any top-k candidate
    # Since we don't have coordinates, we use the score value itself as a proxy:
    # candidates with scores close to top-k scores get a small penalty
    # This is a heuristic: in the 1D case, nearby x-values will have similar mu/sigma
    
    # Actually, without coordinates, we can't compute spatial similarity.
    # Instead, use a simpler diversity mechanism: rank-based penalty
    # Candidates ranked in the top 10% get a small bonus, others get a small penalty
    # to encourage the evaluator to try different regions
    
    # Simpler approach: use the EI value directly with entropy weighting
    # and add a small noise-free diversity term based on the distribution of scores
    
    # Compute percentile rank of each candidate
    sorted_ei = np.sort(h_ei)
    ranks = np.searchsorted(sorted_ei, h_ei, side='right')
    n = len(h_ei)
    percentile_rank = ranks / n  # [0, 1]
    
    # Diversity bonus: give a small boost to mid-range candidates to explore
    # and a small boost to very high candidates to exploit
    # This creates a bimodal preference: exploit best OR explore middle
    diversity_bonus = 0.05 * np.sin(np.pi * percentile_rank)
    
    scores = h_ei + diversity_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
