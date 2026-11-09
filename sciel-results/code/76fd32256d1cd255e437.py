import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Optimistic Expected Improvement (EI) with a constant xi.
    This encourages exploration by requiring a minimum predicted improvement
    to score high, preventing the algorithm from getting stuck in local minima
    where the predicted improvement is small.
    """
    # Safety margin for time, though vectorized ops are fast
    # No complex loop, so deadline guard is minimal
    
    # Optimism parameter: how much better than f_best must the point be?
    # A value of 0.5 is chosen to balance exploration in the multimodal landscape.
    xi = 0.5
    
    # Calculate improvement relative to f_best - xi
    # We want to find points where f(x) < f_best - xi is likely
    # This shifts the threshold, making the acquisition function "optimistic"
    # about the potential gain.
    
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for the optimistic improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # Expected Improvement formula
    # Note: If (f_best - mu - xi) is negative, the CDF term becomes small,
    # but the PDF term still contributes if z is not too negative.
    # However, standard EI is 0 when improvement <= 0.
    # With xi > 0, the "effective best" is f_best - xi.
    # If mu > f_best - xi, the expected improvement over this optimistic threshold is low/zero.
    
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
