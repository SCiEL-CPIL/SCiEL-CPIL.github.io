"""
Strategy: Rich polynomial basis with internal multi-start optimization.
We use a complete polynomial basis up to degree 4 in (x, y) plus some
non-polynomial features (sin, cos, tanh) to capture non-linear structure.
The evaluator's BFGS fits 10 params. We design the equation so that
params map to coefficients of a richer basis, and we also do an internal
refinement using scipy.optimize.minimize on the full parameter set.

Key insight: The evaluator calls equation(x, y, params) with params of length 10.
We can use these 10 params as initial weights for a basis of >10 functions,
then internally re-optimize all basis coefficients using the training data (x, y).
This way, the evaluator's BFGS provides a starting point, and we do our own
optimization inside equation() to get a better fit.

But wait - the evaluator's BFGS is minimizing training MSE by adjusting params.
If we do internal optimization that ignores params, the evaluator's gradient
will be wrong. Instead, we should make params directly control the output in a
way that BFGS can optimize, but with a richer basis.

Better approach: Use params[0:10] as coefficients for a carefully chosen basis
of 10 functions that spans a rich function space. The basis should be designed
so that BFGS starting from all-ones can converge to a good solution.

Let me design a basis of 10 functions:
1. 1 (constant)
2. x
3. y
4. x*y
5. x^2
6. y^2
7. x^2*y
8. x*y^2
9. x^3
10. y^3

This is a polynomial basis up to degree 3 with all terms. For degree 3, we have:
1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3 -> 10 terms exactly!

This is the complete homogeneous polynomial basis of degree 3. It can represent
any cubic polynomial in x and y. This is a natural and complete basis.

However, to capture more complex functions, let me add some non-polynomial terms
by using a different mapping. Actually, let me stick with the complete cubic
polynomial basis since it has exactly 10 terms and is well-conditioned on [-1,1].

To help BFGS converge better, I'll scale the basis functions to have similar
magnitudes. On [-1,1], the polynomial terms are already reasonably scaled.

Let me also consider: the evaluator starts BFGS from all ones. With a cubic
polynomial basis, starting from all ones means the initial prediction is
sum of all basis terms, which is a reasonable starting point.

Let me implement this.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x, y: arrays of input coordinates (same shape)
    params: array of 10 parameters
    Returns: predictions array (same shape as x)
    
    Basis: complete polynomial up to degree 3:
    [1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3]
    """
    # Ensure x, y are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Build the basis matrix (N x 10)
    # Term 0: 1
    # Term 1: x
    # Term 2: y
    # Term 3: x^2
    # Term 4: x*y
    # Term 5: y^2
    # Term 6: x^3
    # Term 7: x^2*y
    # Term 8: x*y^2
    # Term 9: y^3
    
    basis = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x ** 2,
        x * y,
        y ** 2,
        x ** 3,
        x ** 2 * y,
        x * y ** 2,
        y ** 3
    ])
    
    # params should be length 10
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] < 10:
        # Pad if needed
        p = np.pad(p, (0, 10 - p.shape[0]), mode='constant')
    p = p[:10]
    
    # Prediction: basis @ params
    pred = basis @ p
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to simple linear
        pred = params[0] * x + params[1] * y + params[2] if len(params) >= 3 else np.zeros_like(x)
    
    return pred
# EVOLVE-BLOCK-END
