# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools
import math


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya needle triangles using a convex relaxation
    with permutation constraints handled via linear programming.

    The union area is computed by quadrature at 20000 heights. At each height y,
    the cross-section of triangle i is [L_i(y), L_i(y) + w(y)] where
    L_i(y) = x_i + a_i * y and w(y) = (1-y)/64.

    The union length at height y is:
        w + sum_{consecutive pairs in sorted order} min(gap, w)

    We use a piecewise-linear relaxation by introducing auxiliary variables for
    the sorted endpoints and their gaps, then solve a linear program to minimize
    the integral approximation.
    """
    n = 64
    k = int(np.log2(n))
    a = np.array([(i + 1) / n for i in range(n)])
    
    # Number of quadrature points
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    w_vals = (1.0 - y_vals) / n
    
    # L_i(y) = x_i + a_i * y
    # We want to minimize: (1/M) * sum_m [ w_m + sum_{j=0}^{n-2} min(gap_j(m), w_m) ]
    # where gap_j(m) = L_{sigma(j+1)}(y_m) - L_{sigma(j)}(y_m) for some permutation sigma
    
    # This is non-convex due to the min and permutation. We use a relaxation:
    # min(gap, w) <= gap and min(gap, w) <= w
    # So we can upper-bound the union length by: w + sum_j min(gap_j, w)
    # and we want to minimize this upper bound.
    
    # Actually, let's use a different approach: we know that for the Keich construction,
    # the triangles are arranged so that their left endpoints are sorted in a specific way.
    # Let's try to optimize x_i directly using scipy.optimize.
    
    # Define the objective function
    def union_area(x):
        # x: array of shape (n,)
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]  # shape (n, M)
        # Sort L along axis 0
        L_sorted = np.sort(L, axis=0)
        # Compute gaps
        gaps = np.diff(L_sorted, axis=0)  # shape (n-1, M)
        # min(gap, w) for each gap
        min_gaps = np.minimum(gaps, w_vals[np.newaxis, :])
        # Union length at each height
        union_lengths = w_vals + np.sum(min_gaps, axis=0)  # shape (M,)
        # Area is the mean of union lengths
        return np.mean(union_lengths)
    
    # Start from the Keich construction
    positions_keich = []
    j_arr = np.arange(1, k + 1)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich.append(np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k))
    x0 = np.array(positions_keich, dtype=np.float64)
    
    # Use scipy.optimize to minimize the area
    from scipy.optimize import minimize
    
    result = minimize(
        union_area,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    # Use the optimized positions
    positions = result.x
    
    # Ensure all values are finite
    if not np.all(np.isfinite(positions)):
        positions = x0
    
    return positions.astype(np.float64)


# EVOLVE-BLOCK-END
