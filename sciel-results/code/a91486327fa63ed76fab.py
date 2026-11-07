"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis on [-2, 2] with 5 cosines and 5 sines.
    
    Uses frequencies k*pi/2 for k=1..5, giving periods that fit naturally
    into the [-2, 2] domain. The constant term is included as cos(0*x)=1.
    
    Basis: [1, cos(pi*x/2), sin(pi*x/2), cos(pi*x), sin(pi*x),
            cos(3*pi*x/2), sin(3*pi*x/2), cos(2*pi*x), sin(2*pi*x),
            cos(5*pi*x/2), sin(5*pi*x/2)]
    
    This is linear in params, so BFGS should converge reliably.
    The trigonometric basis is orthogonal on symmetric intervals and
    captures both oscillatory and smooth features.
    """
    # 10 basis functions: constant + 5 cos + 5 sin
    # Frequencies: k*pi/2 for k = 0,1,2,3,4,5 with appropriate sin/cos
    # Param 0: constant (cos(0))
    # Params 1-5: cos(k*pi*x/2) for k=1..5
    # Params 6-10... wait, only 10 params total
    
    # Let's use: param0=const, then alternating cos/sin for k=1..5
    # That gives 1 + 5*cos + 5*sin = 11 terms, too many.
    # Instead: 1 constant + 4 cos + 5 sin = 10, or 1 + 5 cos + 4 sin
    # Better: use k=0..4 for cos (5 cos including constant) and k=1..5 for sin (5 sin)
    # Total: 5 cos + 5 sin = 10 params
    
    # cos(k*pi*x/2) for k=0,1,2,3,4 -> 5 cos terms
    # sin(k*pi*x/2) for k=1,2,3,4,5 -> 5 sin terms
    # Total: 10 terms, 10 params
    
    freqs = np.pi / 2.0  # base frequency
    
    # Build basis matrix of shape (n, 10)
    n = len(x)
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Cosine terms: k=0,1,2,3,4
    for k in range(5):
        basis[:, k] = np.cos(k * freqs * x)
    
    # Sine terms: k=1,2,3,4,5
    for k in range(1, 6):
        basis[:, 4 + k] = np.sin(k * freqs * x)
    
    # Linear combination
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
