"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition with adaptive exploration
    # Expected improvement per unit cost, with exploration boost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration: when f_best is close to mu values, explore more
    mu_min, mu_max = np.min(mu), np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    
    # How close is f_best to the current mu range?
    # If f_best is near mu values, we should explore more
    closeness = np.abs(f_best - mu) / (mu_range + 1e-9)
    # Scale exploration weight: higher when f_best is close to mu
    exploration_weight = np.clip(1.0 - closeness, 0.0, 1.0)
    
    # Combine exploitation (EI) and exploration (uncertainty)
    # Exploration term: sigma-based uncertainty
    exploration_term = sigma / (mu_range + 1e-9)
    
    # Balance between exploitation and exploration
    balance = 0.3 + 0.7 * exploration_weight  # 0.3 to 1.0
    
    # Final acquisition: weighted combination per unit cost
    score = (balance * ei + (1 - balance) * exploration_term * (f_best + 1)) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
