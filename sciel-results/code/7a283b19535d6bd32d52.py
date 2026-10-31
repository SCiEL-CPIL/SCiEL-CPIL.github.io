"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for the generated sequences
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0

# Kernel density estimation parameters for empirical PDF of remainders
_KDE_BANDWIDTH = 5.0  # Gaussian kernel bandwidth
_PDF_WEIGHT = 0.5     # Weight for empirical PDF term in combined score

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Dead Space Avoidance with Empirical PDF Guidance.
    
    Combines:
    1. Dead space penalty: r * S(r) where S(r) is Weibull survival function
    2. Linear consolidation bias: beta * r
    3. Empirical PDF guidance: -PDF(r) to favor placements that create
       remainders in high-density regions of the existing gap distribution
    """
    global _START_TIME
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Clip r to non-negative just in case of floating point errors
    r = np.maximum(valid_rems, 0.0)
    
    # Calculate survival probability S(r) = exp(-(r/lambda)^k)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        ratio = r / _WEIBULL_LAMBDA
        power_term = np.power(ratio, _WEIBULL_K)
        survival = np.exp(-power_term)
    
    # Dead space metric: r * S(r)
    dead_space = r * survival
    
    # Linear penalty for consolidation
    beta = 0.1
    linear_penalty = beta * r
    
    # Compute empirical PDF of current bin remainders using Gaussian KDE
    # This guides us to prefer placements that create remainders in
    # high-density regions, promoting consolidation of similar gap sizes
    all_rems = np.maximum(bins, 0.0)
    n_bins = len(all_rems)
    
    if n_bins > 1:
        # Gaussian KDE: PDF(x) = (1/(n*h)) * sum(exp(-(x-xi)^2 / (2*h^2)))
        h = _KDE_BANDWIDTH
        # Vectorized computation for efficiency
        diff = r[:, np.newaxis] - all_rems[np.newaxis, :]
        pdf_values = np.exp(-0.5 * (diff / h) ** 2)
        empirical_pdf = np.sum(pdf_values, axis=1) / (n_bins * h * np.sqrt(2 * np.pi))
    else:
        # If only one bin, no meaningful PDF
        empirical_pdf = np.zeros_like(r)
    
    # Combine penalties: dead_space + linear_penalty - PDF_WEIGHT * empirical_pdf
    # Negative PDF means we prefer high-density regions (consolidation)
    combined_penalty = dead_space + linear_penalty - _PDF_WEIGHT * empirical_pdf
    
    # Score: minimize penalty -> maximize -penalty
    scores[valid] = -combined_penalty
    
    return scores
# EVOLVE-BLOCK-END
