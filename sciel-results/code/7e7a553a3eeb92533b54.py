import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Modified Knowledge Gradient with exploration bonus for minimization.
    
    Combines the expected improvement (KG) with a small exploration term
    to balance exploitation and exploration, especially important in
    small-budget optimization.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values in CDF/PDF calculations
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # KG value: expected improvement over f_best
    # E[max(0, f_best - f(x))] = (f_best - mu) * Phi(z) + sigma * phi(z)
    kg_values = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure non-negative values (numerical stability)
    kg_values = np.maximum(kg_values, 0.0)
    
    # Add a small exploration bonus proportional to posterior uncertainty
    # This helps in early iterations when the model is uncertain
    # The bonus is scaled to be small relative to typical KG values
    exploration_bonus = 0.05 * sigma
    
    # Combine KG with exploration bonus
    scores = kg_values + exploration_bonus
    
    # Check if we're approaching the deadline
    if time.time() > deadline:
        # If we're running out of time, return the scores we have
        # (This should be rare as the computation is fast)
        return scores
    
    return scores
