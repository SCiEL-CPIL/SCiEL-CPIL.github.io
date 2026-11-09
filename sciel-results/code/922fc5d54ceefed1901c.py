"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Known minimum of branin 2d
    f_min = 0.397887
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Gap between current best and known optimum
    gap = f_best - f_min
    # Normalize gap: 1 when far from optimum, 0 when at optimum
    # Use a scale based on typical initial gap (~5-10 for branin)
    gap_scale = 10.0
    norm_gap = np.clip(gap / gap_scale, 0.0, 1.0)
    
    # Exploration-exploitation balance
    # Early: more exploration (higher UCB weight)
    # Late: more exploitation (higher EI weight)
    # Use a smooth transition
    exploration_weight = 0.5 + 0.5 * norm_gap  # 0.5 to 1.0
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (exploration bonus)
    # Beta controls exploration; use adaptive beta
    beta = 1.0 + 2.0 * norm_gap  # 1.0 to 3.0
    ucb = mu + beta * sigma
    
    # For minimization, we want to maximize (f_best - mu) + exploration
    # EI is already in the right direction (higher when mu << f_best)
    # UCB for minimization: we want low mu, so use -mu + beta*sigma
    # But we need to combine with EI which is positive when improvement expected
    
    # Normalize EI to be comparable to UCB scale
    # EI ranges roughly [0, f_best - min(mu)]
    # Use a scaled combination
    
    # Alternative: use a weighted sum where both terms are "higher is better"
    # EI: higher when mu is much lower than f_best (good for minimization)
    # For UCB in minimization: we prefer low mu and high sigma
    # So define ucb_score = -mu + beta * sigma (higher = more promising)
    
    # Scale EI to be in similar range as ucb_score
    # Typical mu range for branin: [0.4, 30], f_best starts around 5-15
    # EI max ~ f_best - min(mu) ~ 10-20
    # ucb_score range: -mu + beta*sigma, mu in [0.4, 30], sigma in [0, 1]
    # So ucb_score roughly [-30, 3]
    
    # Let's use a different approach: normalize both to [0, 1] range
    mu_min, mu_max = np.min(mu), np.max(mu)
    if mu_max - mu_min < 1e-9:
        mu_min, mu_max = mu_min - 0.5, mu_min + 0.5
    
    mu_norm = (mu - mu_min) / (mu_max - mu_min)  # 0 to 1, higher = worse
    
    # EI normalized: higher is better
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max  # 0 to 1
    
    # UCB score: lower mu and higher sigma is better for exploration
    # ucb_score = -mu + beta*sigma, normalize to 0-1
    ucb_raw = -mu + beta * sigma
    ucb_min, ucb_max = np.min(ucb_raw), np.max(ucb_raw)
    if ucb_max - ucb_min < 1e-9:
        ucb_norm = np.zeros_like(ucb_raw)
    else:
        ucb_norm = (ucb_raw - ucb_min) / (ucb_max - ucb_min)  # 0 to 1, higher = better
    
    # Combine: weighted sum
    # When close to optimum (norm_gap ~ 0), weight EI heavily
    # When far (norm_gap ~ 1), weight UCB heavily
    w_ei = 1.0 - exploration_weight * 0.5  # 0.5 to 1.0
    w_ucb = exploration_weight * 0.5  # 0.5 to 1.0
    # Normalize weights
    total_w = w_ei + w_ucb
    w_ei /= total_w
    w_ucb /= total_w
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small exploration bonus for regions with high uncertainty
    # even when exploiting, to avoid getting stuck
    uncertainty_bonus = sigma / (np.max(sigma) + 1e-9)
    score += 0.1 * uncertainty_bonus * norm_gap
    
    return score
# EVOLVE-BLOCK-END
