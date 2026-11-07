"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for computed basis features to avoid recomputation if called multiple times
# with the same input shapes/values (though inputs are fixed per training set)
_basis_cache = {}

def _compute_basis(b, s, temp, pH):
    """
    Compute a fixed set of 10 basis functions.
    These are chosen based on biological priors for E. coli growth:
    1. Bias
    2. Linear b
    3. Linear s
    4. Linear temp
    5. Linear pH
    6. Quadratic temp (to model the peak growth rate)
    7. Quadratic pH (to model pH optimum)
    8. Interaction b*s
    9. RBF for temp centered at 37C (optimal growth temp for E. coli)
    10. Monod-like term for substrate s
    """
    # Check cache
    key = (b.shape, s.shape, temp.shape, pH.shape)
    if key in _basis_cache:
        return _basis_cache[key]

    # Normalize inputs for stability, but keep original scales for specific terms
    # Temp is in [1, 40], pH in [2, 12], b/s in [1, 5]
    
    # 1. Bias
    f0 = np.ones_like(b)
    
    # 2-5. Linear terms
    f1 = b
    f2 = s
    f3 = temp
    f4 = pH
    
    # 6. Quadratic temp
    # Scale temp to center around 0 for better condition number?
    # temp ~ 37 is optimal. Let's use (temp - 20) to keep numbers reasonable.
    t_centered = temp - 20.0
    f5 = t_centered * t_centered
    
    # 7. Quadratic pH
    # pH ~ 7 is optimal.
    p_centered = pH - 7.0
    f6 = p_centered * p_centered
    
    # 8. Interaction b*s
    f7 = b * s
    
    # 9. RBF for temp
    # Center at 37.0, width to cover the range [30, 40] reasonably
    # exp(-((temp - 37)^2) / 20)
    t_diff = temp - 37.0
    f8 = np.exp(-(t_diff * t_diff) / 20.0)
    
    # 10. Monod-like term for substrate
    # 1 / (s + Ks). Let Ks be fixed at 1.0 for now, or use a param?
    # If we use a param, it's non-linear. Let's keep it fixed to ensure linearity in params.
    # The weight for this term will scale it.
    f9 = 1.0 / (s + 1.0)
    
    # Stack into a matrix (N, 10)
    basis_matrix = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # Cache the result
    _basis_cache[key] = basis_matrix
    
    return basis_matrix

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Biologically-informed linear basis model.
    
    y = B @ params
    where B is the fixed basis matrix computed from inputs.
    """
    # Compute the basis matrix
    B = _compute_basis(b, s, temp, pH)
    
    # Ensure params is a column vector or 1D array for dot product
    # params is expected to be length 10
    p = np.asarray(params)
    
    # Clip params to prevent explosion during optimization
    p = np.clip(p, -1e6, 1e6)
    
    # Compute prediction
    y = B @ p
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
