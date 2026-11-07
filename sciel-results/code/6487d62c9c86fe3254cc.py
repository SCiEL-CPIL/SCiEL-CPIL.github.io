import numpy as np
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: Combines Thompson Sampling exploration with
    Expected Improvement exploitation, weighted by exploration budget that
    decays with number of evaluations.
    """
    global _call_count
    _call_count += 1
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Fallback to greedy (maximize -mu) to ensure fast return
        return -mu
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e-8, neginf=1e-8)
    
    # Ensure sigma is non-negative and has a minimum floor
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu(x), sigma(x)^2):
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) 
    #        + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is CDF and phi is PDF of standard normal
    
    # Handle case where sigma is very small
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Thompson Sampling: sample from posterior
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma)
    ts_scores = -samples
    
    # Blend EI and Thompson Sampling
    # Use a weight that starts high for exploration and decreases over time
    # With 30 total evaluations (5 initial + 25 BO steps), we have ~25 acquisition calls
    # Start with 50% exploration, decay to 10% by end
    exploration_weight = max(0.1, 0.5 - 0.02 * _call_count)
    
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-8
    ei_norm = ei / ei_max
    
    ts_max = np.max(ts_scores) if np.max(ts_scores) > 0 else 1e-8
    ts_norm = ts_scores / ts_max
    
    # Combine
    scores = exploration_weight * ts_norm + (1 - exploration_weight) * ei_norm
    
    # Final safety check for finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
