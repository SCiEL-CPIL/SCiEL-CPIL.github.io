import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using POI with uncertainty-gated exploitation refinement.
    
    Primary signal: Probability of Improvement (POI) for exploration.
    Refinement: Small exploitation term that only activates when posterior uncertainty
    is low (sigma below a data-driven threshold), providing fine-grained guidance
    in confident regions without disrupting exploration in uncertain regions.
    """
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return POI quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Primary: POI
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Data-driven threshold: use 25th percentile of sigma as the "confident" threshold
    # This is structural: the threshold adapts to the current posterior spread
    sigma_threshold = np.percentile(sigma_safe, 25)
    
    # Exploitation refinement: only for candidates with low uncertainty
    # Use a small EI-like term, but gated by uncertainty
    # For confident candidates (sigma < threshold), add a small exploitation boost
    # The boost is proportional to how much better the mean is than f_best
    # scaled by a small factor to keep it as a refinement, not a driver
    
    # Compute EI for all, but only use it where sigma is low
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # Gate: only apply EI where sigma is below the threshold
    # Use a smooth gate: gate = 1 where sigma < threshold, 0 otherwise
    # To avoid discontinuities, use a smoothstep-like function
    # gate = clip((threshold - sigma) / threshold, 0, 1)
    gate = np.clip((sigma_threshold - sigma_safe) / max(sigma_threshold, 1e-9), 0.0, 1.0)
    
    # Scale EI to be comparable to POI (which is in [0,1])
    # Normalize EI by its max value to bring it to [0,1] range
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Combine: POI is primary, gated EI is refinement
    # Use a small weight (0.1) for the refinement term
    scores = poi + 0.1 * gate * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
