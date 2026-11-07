"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erfc

# Global state to track evaluation count for adaptive exploration
_eval_count = 0

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
    
    # Strategy: Deterministic Expected Improvement (EI) with a small exploration term.
    # This shifts away from the t-distribution/UCB approach (which can be overly exploratory 
    # or unstable) to a standard, robust EI formulation.
    # The cell finding suggests that "deterministic Expected Improvement plus a small 
    # exploration term (0.1*sigma)" yields higher average regret reduction (1.287 vs 0.9385).
    # 
    # EI for minimization:
    # I(x) = E[max(0, f_best - f(x))]
    # Assuming f(x) ~ N(mu, sigma^2):
    # Let z = (f_best - mu) / sigma
    # I(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Compute z
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) = 0.5 * erfc(-z / sqrt(2))
    # Note: erfc is the complementary error function.
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2))) = 0.5 * erfc(-z / sqrt(2))
    # Using erfc is often more stable for large positive z (where Phi(z) -> 1) and large negative z (where Phi(z) -> 0).
    
    sqrt2 = np.sqrt(2.0)
    
    # Phi(z)
    phi_z = 0.5 * erfc(-z / sqrt2)
    
    # phi(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)
    # We can compute this directly.
    # To avoid underflow for very large z, we can use log-space or just accept 0.
    # For typical values, exp(-z^2/2) is fine.
    phi_pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma * phi_pdf
    
    # Add a small exploration term to prevent pure exploitation and help escape local minima.
    # The finding suggests 0.1 * sigma is a good balance.
    exploration_term = 0.1 * sigma
    
    scores = ei + exploration_term
    
    # Check for non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
    # Check deadline
    if time.time() > deadline:
        # The computation is O(N) and very fast, so this is unlikely to be hit.
        pass

    return scores
# EVOLVE-BLOCK-END
