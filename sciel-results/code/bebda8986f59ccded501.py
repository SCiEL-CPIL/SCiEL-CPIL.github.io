# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining:
    - 5 Chebyshev polynomials (global trend, well-conditioned)
    - 5 Fourier (sin/cos) terms (periodic/quasi-periodic structure)
    
    Total: 10 parameters (linear in params, so BFGS converges well)
    
    Chebyshev basis is mapped to [-1, 1] from [0.1, 6] for stability.
    Fourier terms use frequencies adapted to the domain [0.1, 6].
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # --- Chebyshev Basis (5 terms) ---
    # Map x from [0.1, 6] to [-1, 1]
    x_min, x_max = 0.1, 6.0
    z = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    z = np.clip(z, -1.0, 1.0)
    
    # T0 = 1, T1 = z, T2 = 2z^2 - 1, T3 = 4z^3 - 3z, T4 = 8z^4 - 8z^2 + 1
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * z - 1.0
    T3 = 4.0 * z * z * z - 3.0 * z
    T4 = 8.0 * z**4 - 8.0 * z**2 + 1.0
    
    # --- Fourier Basis (5 terms) ---
    # Use frequencies that capture periodic structures in [0.1, 6]
    # Domain length is ~5.9, so fundamental frequency ~ pi/2.95
    # Use sin and cos at a few different frequencies
    # f1 ~ 1.0, f2 ~ 2.0, f3 ~ 3.0 (radians per unit x)
    # Terms: sin(x), cos(x), sin(2x), cos(2x), sin(3x)
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    sin_3x = np.sin(3.0 * x)
    
    # Stack basis: shape (n, 10)
    # Columns 0-4: Chebyshev, Columns 5-9: Fourier
    basis = np.column_stack([T0, T1, T2, T3, T4, sin_x, cos_x, sin_2x, cos_2x, sin_3x])
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
