"""Combined symbolic regression program: polynomial + Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A flexible model combining polynomial terms (degree 0-4) with Gaussian
    radial basis functions at fixed centers. This provides both global trend
    capture (polynomial) and local feature adaptation (Gaussians), which
    together can approximate a wide range of smooth functions well-conditioned
    for BFGS optimization.
    
    Basis: 1, x, x^2, x^3, x^4, G(x; -1.5), G(x; -0.5), G(x; 0.5), G(x; 1.5),
           sin(x)
    where G(x; c) = exp(-(x-c)^2 / (2*sigma^2)) with sigma=0.6
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Fourier + Polynomial basis
    # This basis is often well-conditioned for BFGS and can approximate
    # a wide range of smooth functions on [-2, 2].
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    
    # Trigonometric terms
    s1 = np.sin(x)
    c1 = np.cos(x)
    s2 = np.sin(2 * x)
    c2 = np.cos(2 * x)
    s3 = np.sin(3 * x)
    c3 = np.cos(3 * x)
    
    # Exponential terms for non-periodic trends
    e1 = np.exp(x)
    e2 = np.exp(-x)
    
    # One Gaussian for local flexibility
    g1 = np.exp(-(x ** 2) / 2.0)
    
    result = (
        p0 * np.ones_like(x) +
        p1 * x +
        p2 * x2 +
        p3 * x3 +
        p4 * s1 +
        p5 * c1 +
        p6 * s2 +
        p7 * c2 +
        p8 * e1 +
        p9 * e2
    )
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
