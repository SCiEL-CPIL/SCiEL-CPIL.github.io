"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge-gradient acquisition function with cost normalization.
    
    Maximizes expected reduction in the minimum observed value,
    accounting for evaluation cost.
    """
    # Track elapsed time to ensure we finish within the deadline
    t0 = time.time()
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Knowledge gradient: approximate expected reduction in f_best
    # The expected new minimum is E[min(f_best, f(x))]
    # = f_best * P(f(x) > f_best) + E[f(x) | f(x) <= f_best] * P(f(x) <= f_best)
    # 
    # For a normal distribution N(mu, sigma^2):
    # P(f(x) <= f_best) = Phi(z) where z = (f_best - mu)/sigma
    # E[f(x) | f(x) <= f_best] = mu - sigma * phi(z)/Phi(z)
    #
    # Expected new minimum:
    # E[min(f_best, f(x))] = f_best * (1 - Phi(z)) + (mu - sigma * phi(z)/Phi(z)) * Phi(z)
    #                      = f_best - f_best*Phi(z) + mu*Phi(z) - sigma*phi(z)
    #                      = f_best - (f_best - mu)*Phi(z) - sigma*phi(z)
    #
    # Expected reduction:
    # f_best - E[min(f_best, f(x))] = (f_best - mu)*Phi(z) + sigma*phi(z) = EI
    #
    # So the knowledge gradient here is equivalent to EI for a single observation.
    # However, we can use a more refined estimate that accounts for the
    # probability that this point actually improves f_best.
    
    # Use a smoothed version that weights by the likelihood of improvement
    # and also considers the magnitude of potential improvement
    
    # Cost-aware: divide by cost to prioritize low-cost evaluations
    # Add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # The acquisition value: EI per unit cost
    # We can also add a small exploration bonus for high uncertainty regions
    # where sigma is large relative to the range of mu
    
    # Compute the acquisition score
    acq = ei / cost_safe
    
    # Add a small exploration term to encourage visiting uncertain regions
    # This helps escape local optima in the acquisition landscape
    # Scale the exploration term so it doesn't dominate
    exploration_bonus = sigma_safe / (np.max(sigma_safe) + 1e-9) * 0.1
    acq = acq + exploration_bonus
    
    # Ensure all values are finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    # Check time - if we're running out, return immediately
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin before 300s limit
        return acq
    
    return acq
# EVOLVE-BLOCK-END
