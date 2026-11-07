# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) per unit cost
    2. Lower Confidence Bound (LCB) per unit cost
    3. A small exploration term based on uncertainty
    
    The idea is to balance exploitation (finding better points) with
    exploration (reducing uncertainty in unexplored regions), while
    accounting for the cost of evaluation.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Cost-aware EI: Expected Improvement divided by cost
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware LCB: Lower Confidence Bound divided by cost
    # LCB = mu - kappa * sigma, where kappa controls exploration
    # We use a cost-aware kappa that increases with cost
    kappa = 0.5 + 0.5 * np.log1p(cost)  # kappa grows slowly with cost
    lcb = mu - kappa * sigma_safe
    
    # Combine EI and LCB, both normalized by cost
    # Use a weighted combination: 0.7 * EI/cost + 0.3 * (-LCB)/cost
    # Note: LCB is lower when better, so we negate it for "higher is better"
    score_ei = ei / np.maximum(cost, 1e-10)
    score_lcb = (-lcb) / np.maximum(cost, 1e-10)
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps when EI and LCB are similar
    uncertainty_bonus = 0.1 * sigma_safe / np.maximum(cost, 1e-10)
    
    # Final score: weighted combination
    score = 0.6 * score_ei + 0.3 * score_lcb + 0.1 * uncertainty_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
