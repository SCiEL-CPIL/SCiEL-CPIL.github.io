"""
Robust symbolic regression via a fixed, well-conditioned feature basis.

Strategy:
- Map x in [-2, 2] to t in [-1, 1].
- Use a basis of 10 features that is universal and well-conditioned:
  Chebyshev polynomials T_0..T_9.
- The evaluator fits 10 parameters via BFGS from ones.
- We return F @ params where F is the (N, 10) feature matrix.

This is a linear model in the features, so BFGS should converge quickly
to the least-squares solution (or close to it), and the orthogonality
of Chebyshev polynomials helps conditioning.

We also add a small amount of regularization-like behavior by ensuring
the basis is numerically stable. No self-optimization of the basis is
done, as that risks local minima and time-outs.

Key improvement over the previous attempt: the previous program had a
score of ~1e11, which suggests it failed catastrophically (possibly due
to a bug or numerical issue). This version is kept simple and robust.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill

def _chebyshev_features(x):
    """
    Compute Chebyshev polynomials T_0(x) ... T_9(x) for x in [-1, 1].
    Returns array of shape (N, 10).
    """
    N = len(x)
    F = np.zeros((N, 10), dtype=float)
    
    # T_0(x) = 1
    F[:, 0] = 1.0
    
    if N > 0:
        # T_1(x) = x
        F[:, 1] = x
        
        # T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
        for n in range(2, 10):
            F[:, n] = 2.0 * x * F[:, n-1] - F[:, n-2]
            
    return F

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
        
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute basis features
    F = _chebyshev_features(t)
    
    # Linear combination
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 10:
        # Fallback if params shape is wrong (shouldn't happen)
        p = np.ones(10)
        
    return F @ p
# EVOLVE-BLOCK-END
