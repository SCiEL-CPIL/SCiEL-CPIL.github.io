# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware, diversity-penalized acquisition function using Matern-5/2 kernel on features.
    
    1. Compute Expected Improvement (EI).
    2. Select top-K candidates by EI.
    3. Compute similarity using Matern-5/2 kernel on standardized (mu, sigma) features.
    4. Penalize candidates based on their cumulative similarity to the top-K set.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # 1. Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Select top-K candidates
    # K=10 seems reasonable for 2000 candidates to keep the penalty set focused
    K = 10
    if n < K:
        K = n
        
    top_k_indices = np.argsort(ei)[-K:]
    
    # 3. Feature Extraction and Standardization
    # Standardize mu and sigma to unit variance
    mu_std = np.std(mu) + 1e-9
    sigma_std = np.std(sigma) + 1e-9
    
    mu_norm = (mu - np.mean(mu)) / mu_std
    sigma_norm = (sigma - np.mean(sigma)) / sigma_std
    
    features_all = np.column_stack((mu_norm, sigma_norm))
    features_top_k = features_all[top_k_indices]
    
    # 4. Matern-5/2 Kernel Similarity
    # Estimate length scale from top-K set distances
    if K > 1:
        diff = features_top_k[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(K, k=1)
        nonzero_dists = dists[iu]
        if len(nonzero_dists) > 0:
            # Use a smaller quantile to make the kernel more local/sensitive
            length_scale = np.percentile(nonzero_dists, 25)
        else:
            length_scale = 1.0
    else:
        length_scale = 1.0
        
    length_scale = max(length_scale, 1e-3)
    
    # Compute distances from all candidates to top-K candidates
    diff_all = features_all[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
    dists_all = np.sqrt(np.sum(diff_all**2, axis=2))
    
    d_scaled = dists_all / length_scale
    
    # Compute Matern-5/2 kernel values
    sqrt3 = np.sqrt(3.0)
    r = d_scaled
    kernel_vals = (1.0 + sqrt3 * r + 0.6 * r**2) * np.exp(-sqrt3 * r)
    
    # 5. Compute Penalties
    # Use SUM of similarities instead of MAX to capture cumulative redundancy
    # This penalizes points that are in a "cluster" of likely picks
    penalties = np.sum(kernel_vals, axis=1)
    
    # 6. Final Score
    # Score = EI - lambda * Penalty
    # Use raw EI to maintain scale consistency
    # Tune lambda to be small enough not to dominate EI, but large enough to encourage diversity
    lambda_penalty = 0.5 * (np.max(ei) / (np.max(penalties) + 1e-9))
    
    scores = ei - lambda_penalty * penalties
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
