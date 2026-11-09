"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Batch-aware greedy diversity-preserving acquisition.
    
    Strategy: Iteratively select top candidates by Expected Improvement,
    then penalize remaining candidates proportional to their similarity
    (via mu correlation) to already-selected points. This spreads out
    exploration across the multimodal Hartmann 3D landscape.
    
    Since we only receive mu and sigma (not coordinates), we use the
    posterior mean values as a proxy for spatial proximity: points with
    similar mu values are likely close in the input space.
    """
    n = len(mu)
    
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] for stable scoring
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Batch-aware greedy selection with diversity penalty
    # We simulate selecting K candidates and assign scores based on
    # when each would be selected (earlier = higher score)
    
    K = 50  # Number of "virtual selections" to spread exploration
    penalty_strength = 0.3  # How strongly to penalize similarity
    
    # Initialize scores with EI
    scores = ei_norm.copy()
    
    # Track selected indices
    selected = []
    remaining = np.arange(n)
    
    # For efficiency, do a simplified version:
    # Instead of full greedy, use a diversity-aware scoring:
    # score_i = ei_norm_i - penalty * max_similarity_to_top_ei_points
    
    # Find top EI points as anchors
    top_ei_idx = np.argsort(ei_norm)[-min(K, n):][::-1]
    
    if len(top_ei_idx) > 0:
        # For each point, compute max similarity to anchor points
        # Similarity via normalized mu correlation (points with close mu are similar)
        mu_centered = mu - np.mean(mu)
        mu_norm = np.sqrt(np.sum(mu_centered**2)) + 1e-9
        
        # Compute similarity matrix between all points and anchors
        # Use cosine similarity on mu vectors (scalar, so just normalized value)
        # Actually for scalars, use absolute difference normalized
        
        anchor_mu = mu[top_ei_idx]
        
        # For each point, find closest anchor in mu-space
        # Use a soft penalty: penalty = penalty_strength * exp(-alpha * |mu_i - mu_anchor|)
        alpha = 10.0  # Controls how local the penalty is
        
        # Compute minimum distance to any anchor
        # mu shape: (n,), anchor_mu shape: (K,)
        # dist[i, j] = |mu[i] - anchor_mu[j]|
        dist = np.abs(mu[:, np.newaxis] - anchor_mu[np.newaxis, :])
        min_dist = np.min(dist, axis=1)
        
        # Penalty: exponential decay with distance
        penalty = penalty_strength * np.exp(-alpha * min_dist)
        
        scores = ei_norm - penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final normalization to [0, 1] for consistency
    s_min = np.min(scores)
    s_max = np.max(scores)
    s_range = s_max - s_min + 1e-9
    scores = (scores - s_min) / s_range
    
    return scores
# EVOLVE-BLOCK-END
