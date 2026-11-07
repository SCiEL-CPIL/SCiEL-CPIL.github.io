"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erfc, erf

# Global state to track evaluation count for adaptive exploration
_eval_count = 0
_TOTAL_EVALS = 30  # Total evaluations including initial 5

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Increment evaluation count
    _eval_count += 1
    
    # Strategy: Adaptive Expected Improvement (EI) with a decaying exploration term.
    # The exploration term helps escape local minima early on, while the EI term
    # focuses on improving the best found value. As evaluations progress, we
    # shift weight from exploration to exploitation.
    
    # Compute z
    z = (f_best - mu) / sigma
    
    sqrt2 = np.sqrt(2.0)
    
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    phi_z = 0.5 * erfc(-z / sqrt2)
    
    # phi(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)
    # Clip z to avoid underflow in exp, though erfc handles large z well.
    # For very large |z|, phi(z) is effectively 0.
    z_clipped = np.clip(z, -500, 500)
    phi_pdf = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma * phi_pdf
    
    # Adaptive exploration coefficient
    # Start with higher exploration, decay to lower exploration.
    # t is 1-based for the first call after init.
    # Progress: 0.0 at start, 1.0 at end.
    progress = min(_eval_count / _TOTAL_EVALS, 1.0)
    
    # Lambda decays from 0.5 to 0.05
    # This ensures we explore broadly initially (high sigma regions)
    # and exploit locally later (low mu regions).
    lambda_explore = 0.5 * (1.0 - progress) + 0.05
    
    scores = ei + lambda_explore * sigma
    
    # Check for non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
    # Check deadline
    if time.time() > deadline:
        # The computation is O(N) and very fast, so this is unlikely to be hit.
        pass

    return scores
# EVOLVE-BLOCK-END
