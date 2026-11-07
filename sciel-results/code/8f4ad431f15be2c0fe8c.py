"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Global cache for the normalized Chebyshev basis matrix
# Keyed by the byte representation of x to handle different x arrays (train/test)
_basis_cache = {}

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Create a cache key based on the data of x
    # Using tobytes is efficient for numpy arrays
    try:
        key = (len(x), x.tobytes())
    except Exception:
        # Fallback if x is not a standard numpy array or has issues
        key = None

    if key is not None and key in _basis_cache:
        T_norm = _basis_cache[key]
    else:
        # Map x from [0.1, 6] to [-1, 1] for Chebyshev basis
        x_min = 0.1
        x_max = 6.0
        # Avoid division by zero if range is 0, though here it is 5.9
        t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
        
        # Precompute Chebyshev polynomials T_0 to T_9
        # T_k(t)
        T = np.zeros((len(x), 10), dtype=np.float64)
        T[:, 0] = 1.0
        T[:, 1] = t
        for k in range(2, 10):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
            
        # Normalize the basis columns to unit norm to improve conditioning
        # for BFGS optimization. This ensures parameters are on a similar scale.
        norms = np.linalg.norm(T, axis=0)
        # Handle zero norms (should not happen for Chebyshev on this interval)
        norms[norms == 0] = 1.0
        T_norm = T / norms
        
        # Cache the result
        if key is not None:
            if len(_basis_cache) > 100:
                _basis_cache.clear()
            _basis_cache[key] = T_norm

    # Compute linear combination
    return T_norm @ params
# EVOLVE-BLOCK-END
