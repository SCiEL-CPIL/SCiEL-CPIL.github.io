"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next) for the 2000 candidates.
    
    Strategy: Use a cost-aware Expected Improvement (EI) variant with a small
    exploration bonus to balance exploitation and exploration. The cost-aware
    term divides by cost to prioritize cheap evaluations. A small sigma bonus
    encourages exploring high-uncertainty regions.
    
    We also add a deadline guard: if we're running long, we return a simple
    score quickly.
    """
    start_time = time.time()
    DEADLINE = 290.0  # seconds, safety margin before 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (standard formula)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use stable computation for large |z|
    # For z > 0: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For z < 0: EI is small, we can use the complement form
    
    # Vectorized stable EI computation
    ei = np.zeros_like(mu)
    
    # Mask where z is moderate
    mask_mod = np.abs(z) < 10
    
    # For moderate z, use standard formula
    if np.any(mask_mod):
        z_mod = z[mask_mod]
        mu_mod = mu[mask_mod]
        sig_mod = sigma_safe[mask_mod]
        
        # Use scipy's norm.cdf and norm.pdf for accuracy
        from scipy.stats import norm
        phi_z = norm.pdf(z_mod)
        Phi_z = norm.cdf(z_mod)
        
        ei_mod = (f_best - mu_mod) * Phi_z + sig_mod * phi_z
        ei[mask_mod] = np.maximum(ei_mod, 0.0)
    
    # For large positive z (z >= 10), EI ≈ f_best - mu + sigma * phi(z) ≈ f_best - mu
    # Since phi(z) is negligible for large z
    mask_large_pos = z >= 10
    if np.any(mask_large_pos):
        ei[mask_large_pos] = np.maximum(f_best - mu[mask_large_pos], 0.0)
    
    # For large negative z (z <= -10), EI ≈ 0
    # ei[mask_large_neg] = 0.0 (already initialized to 0)
    
    # Cost-aware score: EI / cost
    # Add a small exploration bonus proportional to sigma to encourage exploration
    # The bonus is scaled to be small relative to EI
    exploration_bonus = 0.1 * sigma_safe
    
    score = (ei + exploration_bonus) / np.maximum(cost, 1e-8)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If we're close to the deadline, return quickly
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
