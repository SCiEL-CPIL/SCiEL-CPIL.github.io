# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware, diversity-penalized acquisition function using Matern-5/2 kernel.
    
    1. Compute Expected Improvement (EI).
    2. Select top-K candidates by EI.
    3. Compute similarity using Matern-5/2 kernel on standardized (mu, sigma) features.
    4. Penalize candidates based on their maximum similarity to the top-K set.
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
    # K=20 seems reasonable for 2000 candidates
    K = 20
    if n < K:
        K = n
        
    top_k_indices = np.argsort(ei)[-K:]
    
    # 3. Feature Extraction and Standardization
    # Use mu and sigma as features.
    # Standardize to unit variance to make the kernel length scale interpretable.
    mu_std = np.std(mu) + 1e-9
    sigma_std = np.std(sigma) + 1e-9
    
    # Scale mu and sigma
    mu_norm = (mu - np.mean(mu)) / mu_std
    sigma_norm = (sigma - np.mean(sigma)) / sigma_std
    
    features_all = np.column_stack((mu_norm, sigma_norm))
    features_top_k = features_all[top_k_indices]
    
    # 4. Matern-5/2 Kernel Similarity
    # Matern-5/2: (1 + sqrt(3)*d + 3/5*d^2) * exp(-sqrt(3)*d)
    # We need to define a length scale for the kernel in the feature space.
    # Let's estimate it from the top-K set.
    if K > 1:
        # Compute pairwise distances in the 2D feature space for top-K
        # diff[i, j, :] = features_top_k[i] - features_top_k[j]
        diff = features_top_k[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(K, k=1)
        nonzero_dists = dists[iu]
        if len(nonzero_dists) > 0:
            # Use median distance as a heuristic for length scale
            length_scale = np.median(nonzero_dists)
        else:
            length_scale = 1.0
    else:
        length_scale = 1.0
        
    # Ensure length scale is not zero
    length_scale = max(length_scale, 1e-3)
    
    # Compute distances from all candidates to top-K candidates
    # diff_all[i, j, :] = features_all[i] - features_top_k[j]
    diff_all = features_all[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
    dists_all = np.sqrt(np.sum(diff_all**2, axis=2))
    
    # Scale distances by length scale
    d_scaled = dists_all / length_scale
    
    # Compute Matern-5/2 kernel values
    # K(r) = (1 + sqrt(3)*r + 3/5*r^2) * exp(-sqrt(3)*r)
    sqrt3 = np.sqrt(3.0)
    r = d_scaled
    kernel_vals = (1.0 + sqrt3 * r + 0.6 * r**2) * np.exp(-sqrt3 * r)
    
    # 5. Compute Penalties
    # Penalty for candidate i is the max similarity to top-K candidates.
    # For candidates in top-K, exclude self-similarity.
    
    max_sim_all = np.max(kernel_vals, axis=1)
    
    penalties = np.zeros(n)
    
    # For non-top-K candidates, penalty is max similarity to all top-K
    is_top_k = np.zeros(n, dtype=bool)
    is_top_k[top_k_indices] = True
    non_top_k_mask = ~is_top_k
    penalties[non_top_k_mask] = max_sim_all[non_top_k_mask]
    
    # For top-K candidates, exclude self-similarity
    for k_idx, orig_idx in enumerate(top_k_indices):
        row = kernel_vals[orig_idx]
        row_copy = row.copy()
        row_copy[k_idx] = 0.0
        penalties[orig_idx] = np.max(row_copy)
        
    # 6. Final Score
    # Score = EI - lambda * Penalty * EI_Scale
    
    # Normalize EI to [0, 1] to make penalty strength easier to tune
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_max = 1.0
    ei_norm = ei / ei_max
    
    # Lambda: how much to penalize redundancy
    # If lambda is high, we prioritize exploration.
    # If lambda is low, we prioritize exploitation.
    # Since we are doing sequential BO, we want a balance.
    # The penalty is in [0, 1].
    # Let's try lambda = 0.5 to 1.0.
    lambda_penalty = 0.8
    
    scores = ei_norm - lambda_penalty * penalties
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
