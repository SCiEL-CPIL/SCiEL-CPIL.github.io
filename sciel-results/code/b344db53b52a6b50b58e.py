# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-Product Fourier basis with interaction and sum/difference harmonics.
    
    Combines the structural tensor-product idea from the current program with
    the sum/difference harmonic idea from the strongest inspiration.
    
    Basis functions (10 total):
    1. Constant
    2. sin(pi*x)
    3. cos(pi*x)
    4. sin(pi*y)
    5. cos(pi*y)
    6. sin(pi*x)*sin(pi*y)   -- tensor-product interaction
    7. cos(pi*x)*cos(pi*y)   -- tensor-product interaction
    8. sin(pi*(x+y))          -- sum harmonic
    9. cos(pi*(x-y))          -- difference harmonic
    10. sin(2*pi*x)           -- higher frequency for detail
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure 1D
    if x.ndim != 1:
        x = x.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    
    # Precompute trig terms
    sin_px = np.sin(np.pi * x)
    cos_px = np.cos(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_py = np.cos(np.pi * y)
    
    # Tensor-product interactions
    sin_px_sin_py = sin_px * sin_py
    cos_px_cos_py = cos_px * cos_py
    
    # Sum and difference harmonics
    sin_pxy = np.sin(np.pi * (x + y))
    cos_pxy = np.cos(np.pi * (x - y))
    
    # Higher frequency
    sin_2px = np.sin(2 * np.pi * x)
    
    # Build prediction
    pred = (params[0] * 1.0 +
            params[1] * sin_px +
            params[2] * cos_px +
            params[3] * sin_py +
            params[4] * cos_py +
            params[5] * sin_px_sin_py +
            params[6] * cos_px_cos_py +
            params[7] * sin_pxy +
            params[8] * cos_pxy +
            params[9] * sin_2px)
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
