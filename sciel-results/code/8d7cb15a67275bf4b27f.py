"""
Strategy: B-Spline Basis with Fixed Knots.
Instead of global basis functions (like Gaussians or Polynomials) which suffer from
multicollinearity and ill-conditioning that trap BFGS in local minima, we use B-splines.
B-splines have local support and a partition of unity property.
This creates a well-conditioned design matrix where the basis functions are nearly
orthogonal in the L2 sense over the domain.
Since the model is linear in the parameters (coefficients of the splines), BFGS
will effectively solve a linear least squares problem. The well-conditioning ensures
that BFGS converges to the global optimum quickly and reliably, even from the
all-ones initial guess.

We define 10 cubic B-spline basis functions over the domain [0.1, 6.0].
To ensure we have exactly 10 parameters, we use 10 basis functions.
Cubic splines require at least 4 knots. We use a dense set of internal knots
to allow for high-frequency features, but limit the number of basis functions to 10
by choosing the knot vector appropriately.

Actually, simpler: We can use a fixed set of 10 basis functions that are
smooth, local, and well-conditioned.
Let's use a combination of:
1. A constant term (1)
2. 9 cubic B-spline basis functions defined on a uniform grid of knots.

Wait, standard B-spline construction with 10 basis functions of degree 3 requires
n+1 = 11 knots? No, number of basis functions = number of knots - degree - 1.
If we want 10 basis functions of degree 3, we need 14 knots.
Let's just define the basis functions manually using scipy.interpolate.BSpline
or implement a simple local polynomial basis.

A simpler, robust local basis:
Divide [0.1, 6.0] into 10 intervals.
For each interval, define a "bump" function that is non-zero only in that interval
and its neighbors? No, that's just RBFs again.

Let's stick to the mathematically superior B-splines.
We will precompute the B-spline basis matrix values for the training points?
No, the equation function is called for both training and test.
We must compute the basis values on the fly.
Computing 10 cubic B-splines is O(N * 10) which is very fast.

Implementation:
We define a knot vector t.
We want 10 basis functions.
Degree k=3.
Number of knots = n + k + 1 = 10 + 3 + 1 = 14.
Domain [0.1, 6.0].
We can place knots uniformly.
t = np.linspace(0.1, 6.0, 14)
This gives us 10 cubic B-spline basis functions.
The basis functions are:
N_{0,3}(x), ..., N_{9,3}(x).

Properties:
1. Sum of N_{i,3}(x) = 1 for all x in [0.1, 6.0] (Partition of Unity).
2. Each N_{i,3}(x) is >= 0.
3. Support of N_{i,3}(x) is [t_i, t_{i+4}].
4. The design matrix is well-conditioned.

Since the basis sums to 1, the function approximates y as a weighted average of the parameters.
If the target function is smooth, this should fit well.
If the target is high frequency, 10 cubic splines might be underfitting, but they are
much better conditioned than 10 polynomials or 10 Gaussians.

To handle potential high frequency, we could use higher degree splines?
Degree 3 is standard.
Let's try this.

We need to compute B-spline values efficiently.
scipy.interpolate.BSpline.basis_element can be used, but it's slow if called per point.
We can use scipy.interpolate.BSpline.basis(0, t, k, x) which returns the values of all basis functions at x.
This is vectorized and fast.

Check availability: scipy is available.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.interpolate import BSpline

warnings.filterwarnings('ignore')

START_TIME = time.time()
DEADLINE = START_TIME + 140.0

# Precompute knot vector and degree
# Domain: [0.1, 6.0]
# Number of basis functions: 10
# Degree: 3
# Number of knots = 10 + 3 + 1 = 14
t_knots = np.linspace(0.1, 6.0, 14)
degree = 3

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear combination of 10 cubic B-spline basis functions.
    params: [c_0, ..., c_9]
    x: Input array
    """
    x = np.asarray(x).flatten()
    
    # Calculate B-spline basis values
    # BSpline.basis(0, t, k, x) returns an array of shape (n_basis, len(x))
    # We need shape (len(x), n_basis) for dot product with params
    try:
        # basis(0, t, k, x) computes the values of the basis functions
        # Note: The order of basis functions returned matches the knot indices
        B = BSpline.basis(0, t_knots, degree, x)
        # B shape: (10, N)
        # We want pred = B.T @ params
        pred = B.T @ params
    except Exception:
        # Fallback to a simple polynomial if B-spline fails for some reason
        # This should rarely happen
        pred = np.zeros_like(x)
        for i, p in enumerate(params):
            pred += p * (x ** i)
            
    return pred

# EVOLVE-BLOCK-END
