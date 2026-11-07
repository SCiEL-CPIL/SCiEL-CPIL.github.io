"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] != 10:
        p = p[:10] if p.size >= 10 else np.pad(p, (0, 10 - p.size))

    # Check deadline
    if time.time() > deadline:
        return np.zeros(n, dtype=np.float64)

    # Hybrid basis: combine local spline-like flexibility with global features
    # We use 10 basis functions:
    # - 6 shifted cubic B-spline-like basis functions (local, smooth)
    # - 2 trigonometric terms for periodic components
    # - 2 polynomial terms for global trend
    
    # Normalize x to [-1, 1] for numerical stability
    xn = x / 3.0
    
    b = np.ones((10, n), dtype=np.float64)
    
    # Local basis functions: smooth bumps centered at different locations
    # Using a Gaussian-like or cubic B-spline-like shape
    # Centers spread across [-1, 1]
    centers = np.linspace(-1.0, 1.0, 6)
    width = 0.4  # Width of each bump
    
    for i in range(6):
        # Cubic B-spline-like basis: (1 - (x-c)^2/w^2)^3 for |x-c| < w, else 0
        diff = (xn - centers[i]) / width
        b[i, :] = np.where(np.abs(diff) < 1.0, (1.0 - diff**2)**3, 0.0)
    
    # Trigonometric basis for periodic components
    b[6, :] = np.sin(np.pi * xn)
    b[7, :] = np.cos(np.pi * xn)
    
    # Polynomial basis for global trend
    b[8, :] = xn
    b[9, :] = xn**3
    
    # Linear combination
    result = b @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
