"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Kriging Believer (EI-Believer) strategy.
    
    The Kriging Believer approach computes the Expected Improvement (EI) for each
    candidate, then uses the GP mean prediction *conditional on achieving high EI*
    as the score. This is approximated by computing EI and scaling it by a
    cost-aware factor, while adding a small uncertainty bonus to encourage
    exploration in regions with high sigma.
    
    Strategy:
    - Compute EI (Expected Improvement) for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
      where z = (f_best - mu) / sigma
    - Add a small exploration bonus proportional to sigma to avoid premature convergence
    - Scale by 1/cost to prefer cheaper evaluations
    - Return finite scores (higher = evaluate next)
    """
    t0 = time.time()
    DEADLINE = t0 + 280.0  # Safety margin vs 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe cost to avoid division by zero
    safe_cost = np.maximum(cost, 1e-6)
    
    # Compute z for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: when f_best - mu < 0, EI should be 0 (no improvement possible)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    # EI should be non-negative; clip to ensure
    ei = np.maximum(ei, 0.0)
    
    # Kriging Believer: use EI as base, add small exploration bonus
    # The exploration bonus helps in early stages when uncertainty is high
    beta = 0.05  # Small exploration weight
    scores = ei + beta * sigma
    
    # Cost-aware scaling
    scores = scores / safe_cost
    
    # Ensure finite and non-negative
    scores = np.where(np.isfinite(scores) & (scores >= 0), scores, 0.0)
    
    # Deadline guard
    if time.time() > DEADLINE:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Normalize so max is 1 for numerical stability
    mx = np.max(scores)
    if mx > 0 and np.isfinite(mx):
        scores = scores / mx
    
    return scores
# EVOLVE-BLOCK-END
