"""
Strategy: Fixed-Knot B-Spline Basis with Linear Coefficients.

Instead of a fixed linear combination of generic functions (polynomials, trig, etc.) which
leads to an ill-conditioned linear regression problem that BFGS struggles with, we use
a B-Spline basis. B-Splines are locally supported, which means the design matrix is sparse
and well-conditioned. This makes the problem effectively a linear least-squares problem
disguised as a nonlinear one, allowing BFGS to converge quickly to the global optimum
regardless of the starting point (all ones).

We use a moderate number of knots (e.g., 20-30) to ensure high expressiveness while keeping
the condition number manageable. The 10 parameters are the coefficients of 10 selected
B-Spline basis functions (or a subset of a larger set, but here we map 10 params to 10
specific basis functions to keep the interface strict). Actually, to be more robust, we
can use a fixed set of 10 B-Spline basis functions that cover the domain [0.1, 6] well.
Since the input range is fixed, we can precompute the B-Spline basis values for any x
using standard formulas or scipy. However, the task says "standard library, NumPy and
SciPy are available", so we can use scipy.interpolate.BSpline or implement it.
But scipy.optimize.minimize will call equation(x, params) many times.
Implementing B-Spline evaluation efficiently is key.

Let's use a simpler approach: a RBF (Radial Basis Function) network with fixed centers
is also a linear model. But RBFs can be ill-conditioned if centers are close.
B-Splines are better conditioned.

Let's define 10 B-Spline basis functions of order 4 (cubic) over the range [0, 7] (slightly
larger than [0.1, 6]) with uniform knots.
With 10 basis functions, we need enough knots.
Number of knots = n_basis + order = 10 + 4 = 14.
Knots: 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7 (clamped).
This gives 10 basis functions.

The evaluator fits 10 params. We map params[0..9] to the 10 B-Spline coefficients.
The function equation(x, params) computes the linear combination of these 10 B-Splines.

Why this is better:
1. B-Splines form a nearly orthonormal basis on the interval, leading to a well-conditioned
   design matrix.
2. BFGS will converge to the least-squares solution in a few iterations.
3. It can approximate a wide variety of smooth functions.
4. It avoids the numerical instability of high-degree polynomials or ill-conditioned
   exponential/logarithmic mixes.

Implementation details:
- We need to evaluate B-Splines at arbitrary x values.
- We can use scipy.interpolate.BSpline, but it might be slow if called repeatedly for
  different x arrays.
- Alternatively, we can implement the Cox-de Boor recursion directly in NumPy for speed.
- Given the small number of basis functions (10) and points (200), even a simple loop
  or vectorized recursion should be fast enough.

Let's implement the Cox-de Boor recursion for cubic B-Splines.
"""

import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin

# Define knots for cubic B-Splines (order 4)
# We want 10 basis functions.
# n_basis = 10, order = 4 => n_knots = 14
# Domain: [0.1, 6], let's use [0, 7] for safety.
# Uniform interior knots: 1, 2, 3, 4, 5, 6
# Clamped knots at ends: 0,0,0,0 and 7,7,7,7
# Total knots: 0,0,0,0, 1,2,3,4,5,6, 7,7,7,7
KNOTS = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7], dtype=float)
ORDER = 4

def _bspline_basis(x, knots, order):
    """
    Evaluate B-Spline basis functions at points x.
    Returns a matrix of shape (len(x), n_basis).
    Uses Cox-de Boor recursion.
    """
    x = np.asarray(x, dtype=float)
    n_knots = len(knots)
    n_basis = n_knots - order
    
    # Initialize N_0
    # N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
    N = np.zeros((len(x), n_basis), dtype=float)
    
    for i in range(n_basis):
        # Handle the last interval specially if needed, but standard definition uses half-open intervals
        # For clamped knots, the last basis function should include the right endpoint.
        if i < n_basis - 1:
            N[:, i] = ((x >= knots[i]) & (x < knots[i+1])).astype(float)
        else:
            N[:, i] = ((x >= knots[i]) & (x <= knots[i+1])).astype(float)
    
    # Recurse up to order
    for k in range(1, order):
        new_N = np.zeros_like(N)
        for i in range(n_basis - k):
            # Term 1
            denom1 = knots[i+k] - knots[i]
            if denom1 > 1e-12:
                term1 = (x - knots[i]) / denom1 * N[:, i]
            else:
                term1 = 0.0
            
            # Term 2
            denom2 = knots[i+k+1] - knots[i+1]
            if denom2 > 1e-12:
                term2 = (knots[i+k+1] - x) / denom2 * N[:, i+1]
            else:
                term2 = 0.0
            
            new_N[:, i] = term1 + term2
        N = new_N[:, :n_basis-k]
    
    # Pad with zeros to maintain shape (n_basis)
    if N.shape[1] < n_basis:
        N = np.hstack([N, np.zeros((len(x), n_basis - N.shape[1]))])
    
    return N

# Precompute nothing, as x varies. But we can cache the basis calculation if x is repeated?
# The evaluator calls equation with the same x_train during fitting, and x_test at the end.
# We can cache the basis matrix for a given x array.
_basis_cache = {}

def _get_basis(x):
    key = tuple(x)
    if key in _basis_cache:
        return _basis_cache[key]
    B = _bspline_basis(x, KNOTS, ORDER)
    _basis_cache[key] = B
    return B

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a B-Spline basis.
    params: array of 10 coefficients.
    """
    if len(params) != 10:
        # Fallback if params length is wrong (should not happen)
        return np.zeros_like(x)
    
    B = _get_basis(x)
    # B is (n_points, 10), params is (10,)
    # Prediction = B @ params
    pred = B @ params
    return pred

# EVOLVE-BLOCK-START
# The code above is the complete program.
# No additional training is done inside the program; the evaluator's BFGS does the fitting.
# The key is that the function structure (B-Spline linear combination) is well-conditioned
# for BFGS.
# EVOLVE-BLOCK-END
