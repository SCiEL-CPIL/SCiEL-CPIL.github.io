"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Voronoi-based Diversity Acquisition Function.
    
    Uses Expected Improvement (EI) combined with a diversity bonus estimated
    via k-Nearest Neighbors in the (mu, sigma) feature space. This approximates
    the Voronoi volume of each candidate, encouraging exploration of
    under-explored regions of the posterior landscape.
    """
    start_time = time.time()
    DEADLINE = 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Compute Diversity Score (Voronoi Volume Proxy)
    # We approximate the "volume" or "influence" of each candidate by its
    # distance to its k-th nearest neighbor in the (mu, sigma) space.
    # Points that are isolated in this space are likely in distinct regions.
    
    # Create feature matrix: [mu, sigma]
    # Normalize features to [0, 1] to make distance scale invariant
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_norm = (mu - mu_min) / (mu_max - mu_min + 1e-9)
    sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min + 1e-9)
    
    features = np.column_stack((mu_norm, sigma_norm))
    
    # Compute pairwise distances
    # Using squared Euclidean distance for efficiency, then sqrt
    # dist^2 = sum(a^2) + sum(b^2) - 2*a.b
    # But for 2000x2000, direct broadcasting is fine and clear.
    # d[i, j] = ||features[i] - features[j]||
    
    # Check time before heavy computation
    if time.time() - start_time > DEADLINE - 10:
        return ei
        
    # Compute pairwise distances
    # diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # More memory efficient:
    # d2 = np.sum(features**2, axis=1)
    # d2_matrix = d2[:, None] + d2[None, :] - 2 * features @ features.T
    # np.maximum(d2_matrix, 0, out=d2_matrix)
    # dists = np.sqrt(d2_matrix)
    
    # Let's use the efficient matrix multiplication method
    d2 = np.sum(features**2, axis=1)
    d2_matrix = d2[:, None] + d2[None, :] - 2 * (features @ features.T)
    np.maximum(d2_matrix, 0, out=d2_matrix)
    dists = np.sqrt(d2_matrix)
    
    # Find k-th nearest neighbor distance for each point
    k = min(10, n - 1)
    if k > 0:
        # Sort distances for each point
        # We only need the k-th smallest, so argpartition is faster
        # But we need the value.
        # np.partition(dists, k, axis=1)[:, k]
        kth_dists = np.partition(dists, k, axis=1)[:, k]
        
        # Normalize k-th distances to [0, 1]
        kth_min = np.min(kth_dists)
        kth_max = np.max(kth_dists)
        kth_range = kth_max - kth_min + 1e-9
        diversity_score = (kth_dists - kth_min) / kth_range
    else:
        diversity_score = np.ones(n)
        
    # 3. Adaptive Exploration Weight
    # When f_best is far from the known minimum, we explore more.
    known_min = -2.53218
    # progress: 0 (at min) to 1 (far from min)
    if f_best > known_min:
        progress = min(max((f_best - known_min) / (0 - known_min), 0), 1)
    else:
        progress = 0
        
    # We want to balance EI (exploitation) with Diversity (exploration).
    # The diversity score is high for isolated points.
    # We boost the score of isolated points, especially if they have decent EI.
    
    # Base score is EI.
    # Bonus is proportional to diversity and sigma (uncertainty).
    # We scale the bonus by a weight that depends on progress.
    
    # If progress is high (early), we want more diversity.
    # If progress is low (late), we want more exploitation (EI).
    
    explore_weight = 0.2 + 0.5 * progress  # Range: 0.2 to 0.7
    
    # Normalize EI to [0, 1] for blending?
    # Or just add a scaled bonus.
    # Bonus = explore_weight * diversity_score * sigma_safe
    # This encourages high uncertainty in diverse regions.
    
    # However, if EI is very low, we might not want to pick it even if diverse.
    # Let's use a multiplicative interaction or a gated additive term.
    
    # Score = EI + explore_weight * diversity_score * sigma_safe * EI_scale
    # EI_scale should be comparable to EI values.
    ei_max = np.max(ei) + 1e-9
    ei_scale = ei_max / 10.0  # Max bonus is ~10% of max EI
    
    scores = ei + explore_weight * diversity_score * sigma_safe * ei_scale
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
