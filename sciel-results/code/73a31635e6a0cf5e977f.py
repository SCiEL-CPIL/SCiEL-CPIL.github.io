import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) with Exponential Cost Gate.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
       EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma.
       This measures the expected reduction in the objective value.
       
    2. Apply a multiplicative exponential cost gate:
       score = EI * exp(-k * cost)
       
       This is structurally different from additive penalties (like LCB - cost)
       or division (EI / cost). The exponential gate creates a "conditional"
       behavior:
       - For high cost, exp(-k*cost) becomes very small, strongly suppressing
         the score unless EI is huge.
       - For low cost, the gate is closer to 1, allowing EI to dominate.
       
       We need to calibrate 'k'. 
       Cost is in [1, 2].
       If k is too large, we only ever pick the absolute cheapest points (x=0),
       which might not contain the minimum (minimizers are in inner 80%).
       If k is too small, it behaves like EI with a mild penalty.
       
       Let's estimate the scale of EI.
       f_best starts around the mean of cosine8 (approx 0) and decreases.
       The range of cosine8 is [-0.8, 1].
       Standardized targets are used in GP, so mu/sigma are in objective units.
       Typically, EI values in BO are on the order of the scale of the function
       variation, e.g., 0.1 to 1.0.
       
       Let's try k = 1.0.
       exp(-1 * 1) = 0.368
       exp(-1 * 2) = 0.135
       Ratio of gate at cost 2 vs cost 1 is ~0.37.
       This is a significant penalty.
       
       Let's try k = 0.5.
       exp(-0.5 * 1) = 0.607
       exp(-0.5 * 2) = 0.368
       Ratio ~0.6.
       
       Let's try k = 0.25.
       exp(-0.25 * 1) = 0.779
       exp(-0.25 * 2) = 0.607
       Ratio ~0.78.
       
       Given the budget is 30 and we start with 5 points (cost ~1.5 each = 7.5),
       we have ~22.5 budget left.
       Average cost ~1.5 allows ~15 evaluations.
       If we bias towards cheaper points (cost ~1.2), we get ~18 evaluations.
       The gain in number of evaluations might outweigh the loss in spatial coverage.
       
       Let's use k = 0.5 as a moderate gate.
    """
    # Check time
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return np.full_like(mu, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, if mu < f_best, z > 0.
    # Phi(z) is the probability that x < f_best.
    # phi(z) is the PDF.
    
    # To ensure numerical stability, we can use scipy.stats.norm.cdf and pdf
    # But for speed, we might use approximations or just the scipy functions
    # since 2000 points is small.
    
    # However, scipy.stats.norm.cdf can be slow if called element-wise on large arrays?
    # No, it's vectorized.
    
    # Let's compute EI
    # Clip z to avoid overflow in exp if we were doing manual calc, but cdf/pdf handle it.
    # Actually, for large positive z, Phi(z) -> 1, phi(z) -> 0.
    # For large negative z, Phi(z) -> 0, phi(z) -> 0.
    
    # One issue: if mu is very close to f_best, z is small.
    
    # Let's use the standard formula.
    # We want to maximize EI.
    
    # Optimization: 
    # cdf = norm.cdf(z)
    # pdf = norm.pdf(z)
    # ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # This is O(N) with constant factors.
    
    # Let's try to be efficient.
    # norm.cdf and pdf are robust.
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI should be non-negative. Due to floating point, it might be slightly negative.
    # Clip to 0.
    ei = np.maximum(ei, 0.0)
    
    # Apply exponential cost gate
    k = 0.5
    gate = np.exp(-k * cost)
    
    score = ei * gate
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
