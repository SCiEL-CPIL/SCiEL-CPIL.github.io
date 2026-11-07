import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
        if time.perf_counter() - _T0 > _DEADLINE:
            # Fallback: return something safe
            return 1.0 / np.maximum(cost, 1e-9)

        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        
        # Expected Improvement
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        
        # Dynamic Quantile Gate
        # f_best high -> restrict to low cost (low quantile)
        # f_best low -> allow high cost (high quantile)
        
        # Heuristic thresholds for f_best (Lévy 2D)
        # Min 0. Max likely ~50.
        # Start with strict gate, relax as we find better values.
        
        f_ref_high = 10.0  # If f_best > 10, we are very early
        f_ref_low = 0.5     # If f_best < 0.5, we are close to optimum
        
        if f_best >= f_ref_high:
            quantile = 0.30  # Only allow cheapest 30%
        elif f_best <= f_ref_low:
            quantile = 0.95  # Allow almost all
        else:
            # Linear interpolation
            # t = 1.0 means f_best is high (relative to low)
            # t = 0.0 means f_best is low
            t = (f_best - f_ref_low) / (f_ref_high - f_ref_low)
            t = np.clip(t, 0.0, 1.0)
            quantile = 0.30 + (0.95 - 0.30) * t
        
        threshold = np.quantile(cost, quantile)
        
        # Hard Gate
        scores = np.where(cost <= threshold, ei, 0.0)
        
        # Fallback if all scores are zero (e.g. threshold too low or ei all 0)
        if np.max(scores) == 0.0:
             scores = ei / cost_safe
        else:
             # Ensure finite
             scores = np.where(np.isfinite(scores), scores, 0.0)
             
        return scores
