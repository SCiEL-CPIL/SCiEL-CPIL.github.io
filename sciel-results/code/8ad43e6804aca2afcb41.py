import numpy as np
import time

_START_TIME = time.time()
_basis_cache = None
_x_cache = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Basis: Chebyshev (First Kind) + Fourier Series.
    
    Strategy:
    - Combines the structural stability of Chebyshev polynomials (T_0..T_4) 
      with the oscillatory flexibility of a Fourier series (sin/cos terms).
    - This addresses the directive to use a Fourier basis while retaining 
      the well-conditioned polynomial core from the current best design.
    - The basis is linear in parameters, ensuring BFGS convergence to the 
      global minimum for this fixed basis.
    - Caching of the basis matrix speeds up repeated evaluations during 
      optimization.
    
    Basis Functions (10 total):
    1. T_0(z) = 1
    2. T_1(z) = z
    3. T_2(z) = 2z^2 - 1
    4. T_3(z) = 4z^3 - 3z
    5. T_4(z) = 8z^4 - 8z^2 + 1
    6. sin(pi * z)
    7. cos(pi * z)
    8. sin(2*pi * z)
    9. cos(2*pi * z)
    10. sin(3*pi * z)
    
    Note: z is mapped from x in [0.1, 6] to [-1, 1].
    """
    global _basis_cache, _x_cache
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)

    # Time limit guard
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Check cache
    if _x_cache is not None and np.array_equal(x, _x_cache):
        basis = _basis_cache
    else:
        N = len(x)
        if N == 0:
            return np.array([])
            
        # Map x to [-1, 1]
        x_min = 0.1
        x_max = 6.0
        z = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
        
        # Clip to handle any numerical drift, though x is guaranteed in range
        z = np.clip(z, -1.0, 1.0)
        
        basis = np.zeros((10, N), dtype=np.float64)
        
        # --- Chebyshev Polynomials of First Kind (T_0 to T_4) ---
        # T_0 = 1
        basis[0] = 1.0
        
        # T_1 = z
        basis[1] = z
        
        # T_2 = 2z^2 - 1
        basis[2] = 2.0 * z * z - 1.0
        
        # T_3 = 4z^3 - 3z
        basis[3] = 4.0 * z * z * z - 3.0 * z
        
        # T_4 = 8z^4 - 8z^2 + 1
        z2 = z * z
        z4 = z2 * z2
        basis[4] = 8.0 * z4 - 8.0 * z2 + 1.0
        
        # --- Fourier Terms (sin/cos) ---
        # Using pi * z to get one full period over [-1, 1]
        # This allows capturing oscillatory behavior if the target function has it.
        
        # sin(pi * z)
        basis[5] = np.sin(np.pi * z)
        
        # cos(pi * z)
        basis[6] = np.cos(np.pi * z)
        
        # sin(2*pi * z)
        basis[7] = np.sin(2.0 * np.pi * z)
        
        # cos(2*pi * z)
        basis[8] = np.cos(2.0 * np.pi * z)
        
        # sin(3*pi * z)
        basis[9] = np.sin(3.0 * np.pi * z)
        
        _x_cache = x.copy()
        _basis_cache = basis

    # Prediction: y = sum(p_i * basis_i(x))
    # basis shape (10, N), p shape (10,)
    # result shape (N,)
    result = basis.T @ p
    
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
