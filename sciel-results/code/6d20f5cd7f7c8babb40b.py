"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track elapsed time for deadline guard
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Ensure inputs are numpy arrays for vectorized ops
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in norm.cdf/pdf
    # norm.cdf(-10) ~ 0, norm.pdf(-10) ~ 0
    # norm.cdf(10) ~ 1, norm.pdf(10) ~ 0
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (UCB) component
    # Maximizing (beta * sigma - mu)
    # This provides a non-zero signal in regions where EI is zero (explored areas)
    # beta is tuned to balance exploration and exploitation
    beta = 1.5
    ucb = beta * sigma_safe - mu
    
    # Combine EI and UCB
    # We want to leverage the strong exploitation of EI and the global exploration of UCB.
    # A weighted sum is a standard structural combination.
    # w_ei = 1.0, w_ucb = 0.5 seems reasonable given the different scales.
    # UCB can be negative if mu > beta*sigma, which is good (discourages bad areas).
    # EI is always non-negative.
    # To make them comparable, we could normalize, but simple weighting often works.
    # Let's use a blend: score = w_ei * ei + w_ucb * ucb
    # However, the scales might differ. EI is in objective units. UCB is in objective units.
    # Let's try a simple sum first, or perhaps a harmonic mean?
    # Actually, the inspirations used EI * cost_gate.
    # The current program used UCB / cost.
    # Let's combine: (w_ei * EI + w_ucb * UCB) * cost_gate
    
    # Cost-aware penalty
    # Using exponential suppression
    # k=1.0 was used in inspirations 1 and 2.
    # k=1.5 in inspiration 3.
    # Let's stick with k=1.0 as it was in the top performers.
    k = 1.0
    cost_gate = np.exp(-k * cost)
    
    # Combine components
    # We need to be careful with the scale of UCB vs EI.
    # If mu is far from f_best, EI is small, UCB is negative (good).
    # If mu is close to f_best, EI is small, UCB is positive (exploration).
    # This seems complementary.
    
    w_ei = 1.0
    w_ucb = 0.5
    
    scores = (w_ei * ei + w_ucb * ucb) * cost_gate
    
    # Ensure all scores are finite
    # If any are NaN or Inf, replace with 0 (neutral score)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
