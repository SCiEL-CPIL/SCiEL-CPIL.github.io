import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Quantile-rank exploration strategy:
    - Base score: Expected Improvement (EI)
    - Exploration term: rank-based penalty that rewards high-uncertainty candidates
      whose mean is below f_best, using empirical quantile ranks for adaptivity.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Quantile-rank exploration:
    # Rank sigma values from low to high (rank 0 = lowest uncertainty)
    sigma_order = np.argsort(sigma)
    sigma_ranks = np.empty(n, dtype=np.float64)
    sigma_ranks[sigma_order] = np.arange(n, dtype=np.float64)
    
    # Normalize ranks to [0, 1] where 1.0 = highest uncertainty
    sigma_rank_norm = sigma_ranks / max(n - 1, 1)
    
    # Only reward exploration if the mean is promising (below current best)
    promising_mask = mu < f_best
    
    # Exploration score: weighted by rank in top 5% uncertainty AND promising mean
    # Use a smooth transition rather than hard threshold
    # Weight peaks in the top 5% and decays smoothly
    top_pct = 0.05
    rank_threshold = 1.0 - top_pct  # 0.95
    
    # Smooth weight: 0 below threshold, rises to 1 in top 5%
    # Use a sigmoid-like transition for smoothness
    transition_width = 0.02
    exploration_weight = np.where(
        promising_mask,
        np.clip((sigma_rank_norm - (rank_threshold - transition_width)) / transition_width, 0.0, 1.0),
        0.0
    )
    
    # Scale exploration bonus by potential improvement range
    min_mu = np.min(mu)
    range_improvement = max(f_best - min_mu, 1e-6)
    
    # Exploration bonus: proportional to rank weight and improvement potential
    exploration_bonus = 0.15 * range_improvement * exploration_weight
    
    scores = ei + exploration_bonus
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
