"""
Symbolic regression for a nonlinear oscillator using a physics-informed basis.
This design combines a sparse set of physically meaningful terms (damping,
nonlinear restoring force, velocity-dependent corrections) with a small number
of polynomial interaction terms. The basis is designed to capture the typical
structure of nonlinear oscillator dynamics: a restoring force that is odd in x
with possible nonlinear corrections, a damping term that is odd in v, and
cross-terms that allow for more complex behavior.

The key improvements over the pure Chebyshev approach:
1. Physics-informed basis: includes terms like x, x^3, v, x*v, x^2*v, v^3
2. Fewer redundant terms: avoids the ill-conditioning of high-degree polynomials
3. Better conditioning: the terms are chosen to be more linearly independent
   and physically interpretable
"""
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression using a physics-informed basis for a nonlinear oscillator.
    
    Basis functions:
    0: 1 (bias)
    1: x (linear restoring)
    2: v (linear damping)
    3: x^3 (cubic nonlinearity in restoring force)
    4: x*v (velocity-dependent restoring)
    5: v^3 (cubic damping)
    6: x^2 (quadratic correction)
    7: v*x^2 (velocity-dependent quadratic)
    8: x^2*v^2 (quartic interaction)
    9: x*v^2 (velocity-squared modulation of restoring)
    
    This set captures the typical structure of nonlinear oscillator dynamics:
    - Odd terms in x for the restoring force (x, x^3)
    - Odd terms in v for damping (v, v^3)
    - Cross-terms for more complex behavior (x*v, x^2*v, x*v^2, etc.)
    """
    # Compute the basis functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = v
    b3 = x3
    b4 = x * v
    b5 = v3
    b6 = x2
    b7 = v * x2
    b8 = x2 * v2
    b9 = x * v2
    
    # Linear combination
    prediction = (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
    
    return prediction
