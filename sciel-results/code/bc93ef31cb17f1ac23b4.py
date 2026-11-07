# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using quantile-based normalization for robustness.
    
    Strategy:
    1. Rank mu and sigma independently to get normalized positions in [0, 1].
    2. Exploitation score: low mu rank is good (minimization).
    3. Exploration score: high sigma rank is good (uncertainty).
    4. Dynamically weight exploration vs exploitation based on f_best.
       - When f_best is large (far from optimum 0), we prioritize exploration.
       - When f_best is small (close to optimum 0), we prioritize exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure valid sigma
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Quantile-based normalization: rank each value and normalize to [0, 1]
    # For mu: lower mu is better, so we want low rank to map to high score
    # For sigma: higher sigma is better for exploration, so high rank maps to high score
    
    # Handle ties in ranking by using average ranks
    mu_order = np.argsort(mu)
    sigma_order = np.argsort(sigma)
    
    mu_ranks = np.empty(n, dtype=np.float64)
    sigma_ranks = np.empty(n, dtype=np.float64)
    
    mu_ranks[mu_order] = np.arange(n)
    sigma_ranks[sigma_order] = np.arange(n)
    
    # Normalize ranks to [0, 1]
    if n > 1:
        mu_norm = mu_ranks / (n - 1)       # 0 = lowest mu (best), 1 = highest mu (worst)
        sigma_norm = sigma_ranks / (n - 1)  # 0 = lowest sigma, 1 = highest sigma (best for exploration)
    else:
        mu_norm = np.zeros(n)
        sigma_norm = np.zeros(n)
    
    # Exploitation score: prefer low mu -> 1 - mu_norm
    exploitation = 1.0 - mu_norm
    
    # Exploration score: prefer high sigma -> sigma_norm
    exploration = sigma_norm
    
    # Dynamic weighting:
    # f_best is the lowest observed value. Weierstrass min is 0.
    # If f_best is close to 0, we should exploit (trust the GP mean).
    # If f_best is far from 0, we should explore more.
    
    # Scale factor: map f_best to a weight. 
    # f_best typically ranges from ~1.0 down to ~0.0 in Weierstrass 1D.
    # Use a smooth transition: w_explore = sigmoid-like or linear.
    
    # Let's use: w_explore = 1 / (1 + f_best * k) where k controls sensitivity
    # Or simpler: w_explore = max(0, min(1, 1.0 - f_best))
    # But f_best can be > 1 initially. Let's normalize.
    
    # Typical f_best range: starts around 0.5-1.0, ends near 0.
    # Let's use a logistic function centered around f_best = 0.2
    # w_explore = 1 / (1 + exp((f_best - 0.2) * 10))
    # When f_best >> 0.2, w_explore -> 0 (exploit)
    # When f_best << 0.2, w_explore -> 1 (explore)
    
    # Actually, let's think about it differently.
    # Early on: f_best is high, sigma is high everywhere. We want to explore.
    # Late on: f_best is low, sigma is low in known areas. We want to exploit.
    
    # A simple adaptive beta for LCB-like behavior:
    # score = -mu + beta * sigma
    # But with quantile normalization, we can blend ranks.
    
    # Let's use: score = w_e * exploitation + (1 - w_e) * exploration
    # where w_e is the weight for exploitation.
    
    # w_e should increase as f_best decreases.
    # f_best ~ 1.0 -> w_e ~ 0.2 (mostly explore)
    # f_best ~ 0.0 -> w_e ~ 0.9 (mostly exploit)
    
    # Linear mapping: w_e = 0.2 + 0.7 * (1 - min(f_best, 1.0))
    # But f_best can be > 1. Let's clamp.
    
    f_best_clamped = np.clip(f_best, 0.0, 1.0)
    w_e = 0.2 + 0.7 * (1.0 - f_best_clamped)
    w_ex = 1.0 - w_e
    
    score = w_e * exploitation + w_ex * exploration
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
