"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s limit
    
    # Use a blend of Expected Improvement and a mu-rank-biased exploration bonus
    # to adaptively shift the exploration-exploitation trade-off.
    
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute rank-based exploration bonus:
    # Points with lower mu (better predicted value) get higher rank.
    # We use the rank of mu to create a bonus that favors regions with low mu
    # but scaled by sigma to encourage exploration in uncertain areas.
    
    # Get the rank of each mu value (0 = best/lowest mu, 1999 = worst/highest mu)
    mu_ranks = np.argsort(np.argsort(mu))  # rank of each element
    mu_rank_normalized = mu_ranks / (len(mu_ranks) - 1)  # normalize to [0, 1]
    
    # Create an exploration bonus that is high for low mu (good prediction) 
    # and scaled by sigma (uncertainty)
    # We invert the rank so that lower mu (better) gets higher score
    mu_score = 1.0 - mu_rank_normalized  # 1.0 for best mu, 0.0 for worst mu
    
    # Scale the exploration bonus by sigma to encourage exploration in uncertain areas
    exploration_bonus = mu_score * sigma
    
    # Blend EI and exploration bonus
    # The blend weight adapts based on the relative magnitude of sigma
    # When sigma is large, we rely more on exploration; when small, on exploitation
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # Adaptive weight: higher weight for EI when sigma is low (exploit)
    # Use a sigmoid-like function of sigma to blend
    blend_weight = 1.0 / (1.0 + np.exp((sigma - sigma_mean) / (sigma_std + 1e-9)))
    
    # Final score: blend of EI and exploration bonus
    scores = blend_weight * ei + (1.0 - blend_weight) * exploration_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're close to deadline - if so, return immediately
    if time.time() > deadline - 5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
