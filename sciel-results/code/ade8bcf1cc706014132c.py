"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - kappa * sigma; lower LCB means more promising
    # Use adaptive kappa: more exploration when uncertainty is high
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    if max_sigma > 1e-12:
        # kappa scales with relative uncertainty
        kappa = 1.5 + 1.5 * (mean_sigma / max_sigma)
    else:
        kappa = 1.5
    
    lcb = mu - kappa * sigma
    
    # Normalize EI and LCB for fair combination
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    if lcb_max - lcb_min > 1e-12:
        # Higher normalized score for lower LCB
        lcb_norm = 1.0 - (lcb - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # Combine EI and LCB
    weight_ei = 0.6
    weight_lcb = 0.4
    combined = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Rank-based Pareto front selection:
    # Instead of dividing by cost, we use rank-based scoring that enforces
    # a trade-off between improvement potential and cost.
    
    # Compute cost-aware score: we want high combined score AND low cost
    # Use a non-linear transformation that penalizes high cost more heavily
    # cost in [1, 2], so cost - 1 is in [0, 1]
    cost_penalty = 1.0 - (cost - 1.0)  # Higher is better (lower cost)
    
    # Combine with the improvement score using a product (Pareto-like)
    # This enforces that both improvement and low cost matter
    score = combined * cost_penalty
    
    # Additionally, apply a rank-based boost:
    # Candidates that are non-dominated in (score, cost) get a boost
    # Compute rank of each candidate by score
    score_rank = np.argsort(np.argsort(score)) / max(len(score) - 1, 1)
    
    # Compute rank of each candidate by cost (lower cost = higher rank)
    cost_rank = np.argsort(np.argsort(-cost)) / max(len(cost) - 1, 1)
    
    # Pareto front: candidates that are good in both dimensions
    # A candidate is on the Pareto front if no other candidate dominates it
    # (i.e., has both higher score and lower cost)
    # Approximate: use the minimum of the two ranks as a proxy
    pareto_rank = np.minimum(score_rank, cost_rank)
    
    # Final score: blend the direct score with the Pareto-based score
    # This ensures strict trade-off rather than linear division
    final_score = 0.7 * score + 0.3 * pareto_rank
    
    # Ensure finite output
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score
# EVOLVE-BLOCK-END
