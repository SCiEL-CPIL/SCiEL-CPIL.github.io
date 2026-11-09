"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a mixed basis of 10 terms.
    
    Combines Chebyshev polynomials (for smooth polynomial components) with
    trigonometric, hyperbolic, and exponential terms to cover a wider
    hypothesis space. This helps BFGS find good fits for diverse target
    functions, including those with periodic or saturating behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for stability
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Build a pure Chebyshev basis of 10 terms: T_0 to T_9
    # Chebyshev polynomials are orthogonal on [-1, 1], providing a well-conditioned
    # basis for least-squares fitting. This allows BFGS to converge reliably to
    # the optimal polynomial approximation of the target function.
    
    basis = np.empty((10, len(t_safe)), dtype=np.float64)
    basis[0] = np.ones_like(t_safe)
    basis[1] = t_safe.copy()
    
    # Recurrence relation: T_{n+1}(t) = 2t * T_n(t) - T_{n-1}(t)
    for k in range(2, 10):
        basis[k] = 2.0 * t_safe * basis[k-1] - basis[k-2]
    
    # Build the linear combination
    result = np.zeros_like(t_safe)
    for k in range(10):
        result += params[k] * basis[k]
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
