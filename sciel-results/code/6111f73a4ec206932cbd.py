"""Symbolic regression with 2 inputs using a rich polynomial basis + BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _build_basis(x, y, degree=4):
    """Build a polynomial basis up to given degree, including interactions."""
    terms = []
    # Constant
    terms.append(np.ones_like(x))
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            terms.append(x ** i * y ** j)
    return np.column_stack(terms)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin before 120s limit

    # Build basis matrix
    basis = _build_basis(x, y, degree=4)
    # basis shape: (n_points, n_terms)

    # We have 10 params. Use them as coefficients for the first 10 basis terms,
    # but actually we can use all basis terms if we map params cleverly.
    # Better: use all basis terms, but we only have 10 params.
    # Strategy: use a subset of the most informative terms, or use all terms
    # and let BFGS fit 10 params that combine them.

    # Actually, let's use a different approach: use all polynomial terms up to degree 3
    # and select which 10 to use based on data.
    
    # Alternative: use a fixed set of 10 terms that covers a good range
    # Terms: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2
    # That's 10 terms for degree up to 3.
    
    # Let's build exactly 10 terms
    terms = []
    terms.append(np.ones_like(x))          # 0: constant
    terms.append(x)                         # 1: x
    terms.append(y)                         # 2: y
    terms.append(x ** 2)                    # 3: x^2
    terms.append(y ** 2)                    # 4: y^2
    terms.append(x * y)                     # 5: xy
    terms.append(x ** 3)                    # 6: x^3
    terms.append(y ** 3)                    # 7: y^3
    terms.append(x ** 2 * y)                # 8: x^2*y
    terms.append(x * y ** 2)                # 9: x*y^2
    
    basis = np.column_stack(terms)  # shape (n, 10)
    
    # Simple linear model: prediction = basis @ params
    # This is linear in params, so BFGS should converge quickly
    return basis @ params


# EVOLVE-BLOCK-END
