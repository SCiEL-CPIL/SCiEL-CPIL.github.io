"""Symbolic regression with a well-conditioned Fourier + polynomial basis."""
import numpy as np
import time

# Pre-compute nothing here; everything is done per-call but efficiently.
# We design a basis that is smooth, bounded, and linearly independent,
# so BFGS from all-ones converges reliably.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using 10 parameters.
    
    Basis design:
    - x is in [0.1, 6]. We map to t = (x - 3.05) / 2.95 * pi  -> roughly [-pi, pi]
      Actually let's map [0.1, 6] to [0, 2*pi] for full period coverage:
      t = (x - 0.1) / 5.9 * 2 * pi
    - We use a mix of constant, linear, and low-frequency trig terms.
    
    Basis functions (10 total):
    b0 = 1
    b1 = x
    b2 = sin(t)
    b3 = cos(t)
    b4 = sin(2*t)
    b5 = cos(2*t)
    b6 = sin(3*t)
    b7 = cos(3*t)
    b8 = x**2 / 36   (scaled to be O(1))
    b9 = x**3 / 216  (scaled to be O(1))
    
    This gives BFGS a smooth, well-conditioned landscape.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to [0, 2*pi]
    t = (x - 0.1) / 5.9 * (2.0 * np.pi)
    
    # Build basis
    b0 = np.ones_like(x)
    b1 = x
    b2 = np.sin(t)
    b3 = np.cos(t)
    b4 = np.sin(2.0 * t)
    b5 = np.cos(2.0 * t)
    b6 = np.sin(3.0 * t)
    b7 = np.cos(3.0 * t)
    b8 = x**2 / 36.0
    b9 = x**3 / 216.0
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
