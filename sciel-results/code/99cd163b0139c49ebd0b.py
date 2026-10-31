"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats as stats

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _estimate_weibull_params_fast(data):
    """
    Estimate Weibull shape k and scale lambda from data mean and variance.
    Returns (k, lambda).
    """
    n = len(data)
    if n < 2:
        return 1.0, 1.0
    
    mean = np.mean(data)
    var = np.var(data)
    
    if mean <= 1e-9 or var <= 1e-9:
        return 1.0, mean if mean > 1e-9 else 1.0
    
    cv = np.sqrt(var) / mean
    
    # Invert CV to find k
    # CV(k) = sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2) / Gamma(1+1/k)
    # We use a pre-computed lookup or fast bisection.
    # Since we are in a loop, let's use a very fast approximation or cached lookup.
    # For speed, we'll use a simple bisection but limit iterations.
    
    def cv_weibull(k):
        try:
            g1 = stats.gamma(1 + 1/k).mean()
            g2 = stats.gamma(1 + 2/k).mean()
            var_term = g2 - g1**2
            if var_term < 0:
                return 0
            return np.sqrt(var_term) / g1
        except:
            return 0
    
    # Bisection
    lo, hi = 0.5, 10.0
    for _ in range(20):
        mid = (lo + hi) / 2
        cv_mid = cv_weibull(mid)
        if cv_mid > cv:
            lo = mid
        else:
            hi = mid
    
    k = (lo + hi) / 2
    g1 = stats.gamma(1 + 1/k).mean()
    lambda_ = mean / g1
    
    return k, lambda_

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    n = len(bins)
    scores = np.full(n, -1e9)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    valid_indices = np.where(valid)[0]
    valid_bins = bins[valid_indices]
    remainders = valid_bins - item
    
    # Strategy: Adaptive Weibull-weighted Best Fit
    # 1. Best Fit component: -remainders
    # 2. Weibull component: bonus for remainders that are "likely" item sizes
    #    (High PDF). This helps avoid creating remainders that are "awkward".
    
    # Estimate Weibull params from ACTIVE bins (0 < bin < 100)
    # Using all bins includes many 100s (empty) and 0s (full), which skews stats.
    active_mask = (bins > 1e-9) & (bins < 100 - 1e-9)
    if np.sum(active_mask) >= 5:
        active_bins = bins[active_mask]
        k, lambda_ = _estimate_weibull_params_fast(active_bins)
    else:
        # Default params if not enough data
        k, lambda_ = 1.5, 20.0 # Typical for capacity 100 Weibull items
        
    # Compute Weibull PDF for remainders
    pdf_values = np.zeros_like(remainders)
    mask_pos = remainders > 1e-9
    if np.any(mask_pos):
        x = remainders[mask_pos]
        # f(x) = (k/lambda_) * (x/lambda_)^(k-1) * exp(-(x/lambda_)^k)
        # Clip x to avoid overflow/underflow issues if lambda is small
        x_safe = np.clip(x, 1e-6, 100.0)
        pdf_values[mask_pos] = (k / lambda_) * (x_safe / lambda_) ** (k - 1) * np.exp(-(x_safe / lambda_) ** k)
    
    # Normalize PDF to [0, 1]
    max_pdf = np.max(pdf_values)
    if max_pdf > 1e-9:
        pdf_norm = pdf_values / max_pdf
    else:
        pdf_norm = pdf_values
    
    # Adaptive Alpha
    # For large items, Best Fit is very strong. Weibull is less important.
    # For small items, we might want to be more careful about fragmentation.
    # Let's make alpha small for large items and slightly larger for small items.
    # item_ratio = item / 100.0
    # alpha = 0.1 * (1.0 - item_ratio)  # More Weibull weight for small items?
    # Actually, let's keep it simple and small to not disrupt Best Fit.
    alpha = 0.05
    
    # Score = -remainders + alpha * pdf_norm
    scores_valid = -remainders + alpha * pdf_norm
    
    scores[valid_indices] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
