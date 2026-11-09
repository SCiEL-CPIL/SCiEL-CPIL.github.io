"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a 10-term Fourier (trigonometric) basis on [0.1, 6].
    
    Basis functions:
      1, cos(k*pi*x/L), sin(k*pi*x/L) for k=1..4
    where L = 5.9 (approx domain length) to make frequencies well-conditioned.
    
    This gives 1 + 4 + 4 = 9 basis functions. We use 10 params by adding a
    constant offset term explicitly or using k=1..5 with one less sine.
    
    Actually, let's use:
      1, cos(w_k * x), sin(w_k * x) for k=1..4
    That's 9 terms. For 10 params, add cos(w_5 * x) without sin(w_5 * x),
    or use k=1..5 with 5 cos and 5 sin = 11... too many.
    
    Let me use: 1, cos(w_k*x), sin(w_k*x) for k=1..4 gives 9.
    Add one more: x/L as a linear trend? Or use 5 cos and 5 sin = 11 terms?
    
    Better: use 10 terms = 1 + 4*cos + 4*sin + 1*linear
    Or simply: 5 cos terms and 5 sin terms (no explicit constant, since cos(0)=1)
    
    Let's do: cos(w_k * x) and sin(w_k * x) for k=0..4, where w_0=0
    cos(0*x)=1, sin(0*x)=0. So that gives 1 + 4 cos + 4 sin = 9. Still 9.
    
    For exactly 10 params: use k=1..5 for cos (5 terms) and k=1..5 for sin (5 terms) = 10.
    No explicit constant, but the DC component will be captured by the combination.
    """
    x = np.asarray(x, dtype=float)
    
    # Domain is [0.1, 6], length ~5.9
    # Use frequencies w_k = k * pi / L where L = 5.9
    L = 5.9
    n_terms = 5  # 5 cos + 5 sin = 10 params
    
    # Precompute frequencies
    k = np.arange(1, n_terms + 1)  # 1, 2, 3, 4, 5
    w = k * np.pi / L  # angular frequencies
    
    # Build basis matrix: 10 x N
    # First 5 rows: cos(w_k * x) for k=1..5
    # Last 5 rows: sin(w_k * x) for k=1..5
    N = len(x)
    basis = np.empty((10, N), dtype=float)
    
    for i, wk in enumerate(w):
        basis[i] = np.cos(wk * x)
        basis[n_terms + i] = np.sin(wk * x)
    
    # Combine with params
    return params @ basis
# EVOLVE-BLOCK-END
