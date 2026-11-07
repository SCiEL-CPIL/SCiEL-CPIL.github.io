"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Strategy: Cost-aware Expected Improvement (EI) with a small exploration bonus.
    We compute EI / cost, but also add a small term that rewards high uncertainty
    relative to the current best, to ensure exploration in the early stages.
    
    The key insight is that Rastrigin is multimodal, so pure exploitation (EI)
    can get stuck in local minima. A small UCB-like term helps escape.
    """
    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI: EI / cost
    score_ei_cost = ei / cost
    
    # Add a small exploration bonus: -kappa * sigma / cost
    # This encourages exploring high-uncertainty regions
    # kappa is small to not dominate the EI term
    kappa = 0.5
    exploration_bonus = -kappa * sigma / cost
    
    # Combine: EI/Cost + exploration bonus
    # The exploration bonus is negative, so it reduces the score for high sigma
    # Wait, that's wrong. We want to INCREASE the score for high sigma.
    # So we should ADD a positive term for high sigma.
    # Let's use: EI/Cost + kappa * sigma / cost
    # But this might over-explore. Let's use a smaller kappa.
    kappa = 0.1
    score = score_ei_cost + kappa * sigma / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
