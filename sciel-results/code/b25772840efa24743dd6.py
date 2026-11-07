import numpy as np
import time
from scipy.stats import norm

_T0 = None
_DEADLINE = None

def _ensure_deadline():
    global _T0, _DEADLINE
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 295.0  # 5s margin for safety
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    elapsed = time.perf_counter() - _T0
    
    # Fallback if time is running out
    if elapsed > _DEADLINE - 5.0:
        # Simple LCB fallback (higher score = more exploration/exploitation mix)
        # Using beta=1.0 as a standard balance
        return -mu + 1.0 * sigma

    # Handle sigma == 0 to avoid division by zero
    # If sigma is 0, EI is deterministic
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Compute base Z for standard EI (xi=0)
    # Z = (f_best - mu) / sigma
    Z_base = (f_best - mu) / sigma_safe
    
    # Standard EI (xi=0)
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    phi_base = norm.pdf(Z_base)
    Phi_base = norm.cdf(Z_base)
    ei_standard = (f_best - mu) * Phi_base + sigma_safe * phi_base
    
    # Multi-Xi EI Approach
    # We evaluate EI for different xi (improvement thresholds)
    # xi > 0: more exploitative (requires larger improvement)
    # xi < 0: more exploratory (counts potential improvements)
    
    # Scale xi based on the uncertainty of the current batch to be adaptive
    # Using the mean sigma of the candidates as a scale reference
    sigma_mean = np.mean(sigma_safe)
    
    # Define a set of xi multipliers relative to sigma_mean
    # These correspond to different levels of optimism/pessimism
    xi_multipliers = [0.0, 0.5, 1.0, -0.5, -1.0, 2.0]
    
    best_score = ei_standard.copy()
    
    for mult in xi_multipliers:
        if mult == 0.0:
            continue # Already computed
        
        xi = mult * sigma_mean
        
        # Z_xi = (f_best - mu - xi) / sigma
        Z_xi = (f_best - mu - xi) / sigma_safe
        
        # EI_xi = (f_best - mu - xi) * Phi(Z_xi) + sigma * phi(Z_xi)
        # Note: If f_best - mu - xi is negative, the term (f_best - mu - xi) * Phi(Z_xi) is negative,
        # but EI is defined as E[max(f_best - f - xi, 0)].
        # The formula holds for all Z.
        
        phi_xi = norm.pdf(Z_xi)
        Phi_xi = norm.cdf(Z_xi)
        
        ei_xi = (f_best - mu - xi) * Phi_xi + sigma_safe * phi_xi
        
        # Update best score element-wise
        best_score = np.maximum(best_score, ei_xi)
        
        # Time check inside loop (optional but safe)
        if time.perf_counter() - _T0 > _DEADLINE - 1.0:
            break

    # Ensure finite values
    best_score = np.where(np.isfinite(best_score), best_score, 0.0)
    
    return best_score
