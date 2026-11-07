"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track evaluation count for adaptive scheduling
_eval_count = 0
_start_time = time.time()
_TIMEOUT = 290.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Normalized Expected Improvement (nEI) 
    with adaptive exploration decay.
    
    Args:
        mu: GP mean at 2000 candidate points (shape: 2000,)
        sigma: GP standard deviation at 2000 candidate points (shape: 2000,)
        f_best: Lowest observed value so far
    
    Returns:
        scores: Array of 2000 finite scores, higher is better.
    """
    global _eval_count, _start_time

    # Deadline guard
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Greedy exploitation (minimize mu)
        return -mu

    # Increment evaluation count
    _eval_count += 1
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Normalized Expected Improvement (nEI)
    # For minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # nEI = EI / (sigma + epsilon) = (f_best - mu)/sigma * Phi(z) + phi(z)
    
    # Avoid division by zero in z calculation, though sigma is maxed at 1e-8
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    # Compute nEI
    # Note: norm.cdf(z) and norm.pdf(z) are efficient in scipy
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    n_ei = z * cdf_z + pdf_z
    
    # Adaptive Exploration Weight
    # We want to weight the uncertainty (sigma) more heavily early on.
    # nEI already incorporates sigma, but we can blend with a pure uncertainty term (UCB-like)
    # that decays over time.
    
    # Total evaluations allowed: 30. Initial: 5. Acquisition calls: 25.
    # Let's define a decay schedule based on _eval_count.
    # Early: high exploration. Late: high exploitation.
    
    # A simple decay: alpha starts at 1.0 and decays to 0.1 by the end.
    # Or use a cosine decay for smoother transition.
    # progress = _eval_count / 25.0
    # alpha = 0.1 + 0.9 * np.cos(0.5 * np.pi * progress)
    
    # Let's use a linear decay for simplicity and predictability, 
    # but ensure it doesn't go to 0 completely to maintain some exploration.
    # Actually, nEI is a very strong acquisition function. 
    # Pure nEI is often sufficient. 
    # However, to make it "structurally different" and potentially better for multimodal Weierstrass,
    # we can blend nEI with a "Probability of Improvement" (PI) term or just use nEI.
    # nEI is generally superior to EI for BO.
    
    # Let's stick to pure nEI but scale it slightly if we want to encourage more exploration early.
    # Actually, nEI naturally favors points with high uncertainty and high mean improvement.
    # The issue with TS was randomness. nEI is deterministic.
    
    # To add a structural twist: "Adaptive nEI"
    # We can scale the acquisition by a factor that depends on the relative uncertainty.
    # But let's first try pure nEI, which is a standard, robust choice.
    # The current program is TS. Switching to deterministic nEI is a major structural change.
    
    # One potential issue: if f_best is very low, and mu is similar, z is small.
    # nEI handles this well.
    
    scores = n_ei
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
