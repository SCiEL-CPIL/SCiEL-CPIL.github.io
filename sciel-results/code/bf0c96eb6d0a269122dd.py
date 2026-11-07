"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute a lookup table for norm.pdf and norm.cdf for speed
_N = 100001
_z_min, _z_max = -10.0, 10.0
_z_grid = np.linspace(_z_min, _z_max, _N)
# We will compute these on first use or precompute
_phi_cache = None
_F_cache = None

def _ensure_caches():
    global _phi_cache, _F_cache
    if _phi_cache is None:
        # Use scipy.stats.norm for accuracy, or manual? 
        # Standard library + NumPy + SciPy available.
        from scipy.stats import norm
        _phi_cache = norm.pdf(_z_grid)
        _F_cache = norm.cdf(_z_grid)

def _norm_pdf(z):
    _ensure_caches()
    # Clip z to valid range
    z_clipped = np.clip(z, _z_min, _z_max)
    # Linear interpolation
    idx = (z_clipped - _z_min) / (_z_max - _z_min) * (_N - 1)
    idx = np.clip(idx, 0, _N - 1)
    i0 = idx.astype(int)
    i1 = np.minimum(i0 + 1, _N - 1)
    frac = idx - i0
    return _phi_cache[i0] * (1 - frac) + _phi_cache[i1] * frac

def _norm_cdf(z):
    _ensure_caches()
    z_clipped = np.clip(z, _z_min, _z_max)
    idx = (z_clipped - _z_min) / (_z_max - _z_min) * (_N - 1)
    idx = np.clip(idx, 0, _N - 1)
    i0 = idx.astype(int)
    i1 = np.minimum(i0 + 1, _N - 1)
    frac = idx - i0
    return _F_cache[i0] * (1 - frac) + _F_cache[i1] * frac


def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI using cached functions for speed
    phi_z = _norm_pdf(z)
    cdf_z = _norm_cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    # Deadline guard: if we are taking too long, just return EI.
    # The computation above is vectorized and should be very fast (<10ms for 2000 points).
    # No complex optimization needed inside acquisition.
    
    return ei
# EVOLVE-BLOCK-END
