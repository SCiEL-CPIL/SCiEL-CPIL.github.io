"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model: quadratic/cubic polynomial + 2 learnable Gaussian bumps.
    
    Structural change from linear-in-params basis to a model where parameters
    control the *shape* of the function, allowing BFGS to search over a richer
    hypothesis space (functional forms, not just linear combinations).
    
    Model:
      f(x) = a0 + a1*x + a2*x^2 + a3*x^3
             + A1 * exp(-(x - c1)^2 / (2*w1^2))
             + A2 * exp(-(x - c2)^2 / (2*w2^2))
    
    Params (10):
      p[0] = a0 (constant)
      p[1] = a1 (linear)
      p[2] = a2 (quadratic)
      p[3] = a3 (cubic)
      p[4] = A1 (Gaussian 1 amplitude)
      p[5] = c1 (Gaussian 1 center)
      p[6] = lw1 (log-width for Gaussian 1, w1 = exp(lw1))
      p[7] = A2 (Gaussian 2 amplitude)
      p[8] = c2 (Gaussian 2 center)
      p[9] = lw2 (log-width for Gaussian 2, w2 = exp(lw2))
    
    Widths use exp() to ensure positivity. Centers are unconstrained.
    This allows BFGS to discover localized features, peaks, or dips that
    a fixed linear basis cannot represent.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Polynomial terms
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3
    
    # Gaussian 1
    A1 = p[4]
    c1 = p[5]
    lw1 = p[6]
    # Clamp log-width to avoid overflow/underflow
    lw1 = np.clip(lw1, -5.0, 5.0)
    w1 = np.exp(lw1)
    g1 = A1 * np.exp(-(x - c1)**2 / (2.0 * w1**2))
    
    # Gaussian 2
    A2 = p[7]
    c2 = p[8]
    lw2 = p[9]
    lw2 = np.clip(lw2, -5.0, 5.0)
    w2 = np.exp(lw2)
    g2 = A2 * np.exp(-(x - c2)**2 / (2.0 * w2**2))
    
    result = poly + g1 + g2
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
