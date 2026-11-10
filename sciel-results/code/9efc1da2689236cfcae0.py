# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya needle triangles using projected
    subgradient descent on the convex piecewise-linear objective.
    """
    n = 64
    M = 20000
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64

    # Precompute heights
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    w = (1.0 - y) / n             # shape (M,)

    # L_i at height y: L[i, m] = x[i] + a[i] * y[m]
    # We'll work with the matrix A where A[i, m] = a[i] * y[m]
    A = np.outer(a, y)  # shape (n, M)

    def compute_grad(x):
        """Compute subgradient of the mean union length w.r.t. x."""
        # L[i, m] = x[i] + A[i, m], shape (n, M)
        L = x[:, None] + A  # (n, M)

        # Sort L along axis 0 at each height
        order = np.argsort(L, axis=0)  # (n, M)

        # Union length: w + sum over consecutive pairs of min(gap, w)
        # gap[j, m] = L[order[j+1, m], m] - L[order[j, m], m]
        L_sorted = L[order, np.arange(M)[None, :]]  # (n, M)

        gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
        overlaps = np.minimum(gaps, w[None, :])     # (n-1, M)

        # Subgradient w.r.t. x:
        # For each height m, the subgradient of union length w.r.t. x_i:
        #   +1 for the smallest L (first in sorted order)
        #   -1 for the largest L (last in sorted order)
        #   For each pair j where gap < w: +1 to the left element, -1 to the right
        #
        # So for element at sorted position p (0-indexed):
        #   contribution = +1 (if p==0) - 1 (if p==n-1)
        #                 + 1 (if p > 0 and gap[p-1] < w)  [it's the right of pair p-1]
        #                 - 1 (if p < n-1 and gap[p] < w)   [it's the left of pair p]
        #
        # Wait, let me re-derive. The union length as a function of the sorted L values:
        # U = w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
        #
        # dU/dL_j:
        #   From min(L_{j+1} - L_j, w): if gap < w, derivative w.r.t. L_j is -1, w.r.t. L_{j+1} is +1
        #   From min(L_j - L_{j-1}, w): if gap < w, derivative w.r.t. L_j is +1, w.r.t. L_{j-1} is -1
        #
        # So for sorted position j:
        #   dU/dL_j = [1 if j>0 and gap[j-1]<w else 0] - [1 if j<n-1 and gap[j]<w else 0]
        #
        # But U is also a function of x_i through L_{pi(j), m} = x_{pi(j), m} + A_{pi(j), m}
        # So dU/dx_i = sum over heights m where i is at sorted position j: dU/dL_j
        #
        # Actually, since we're summing over all heights and the sorting depends on x,
        # the subgradient is: for each height m, for each i, the coefficient is
        #   c_i = [1 if i is at sorted pos j>0 and gap[j-1]<w] - [1 if i is at sorted pos j<n-1 and gap[j]<w]
        #
        # Let me compute this directly.

        grad = np.zeros((n, M))  # will accumulate per-height subgradients

        # For each sorted position j, the subgradient contribution to L at that position
        # pos_grad[j, m] = [1 if j>0 and gap[j-1,m]<w[m]] - [1 if j<n-1 and gap[j,m]<w[m]]

        pos_grad = np.zeros((n, M))
        # For j >= 1: +1 if gap[j-1] < w
        pos_grad[1:, :] += (gaps < w[None, :]).astype(float)  # gap[j-1] for position j
        # For j <= n-2: -1 if gap[j] < w
        pos_grad[:-1, :] -= (gaps < w[None, :]).astype(float)  # gap[j] for position j

        # Now scatter: for each height m, the element at sorted position j gets pos_grad[j, m]
        # So grad[i, m] = pos_grad[order[i, m], m]
        # Using advanced indexing:
        grad = pos_grad[order, np.arange(M)[None, :]]  # (n, M)

        # Average over heights
        return grad.mean(axis=1)

    def compute_area(x):
        """Compute the area (mean union length)."""
        L = x[:, None] + A  # (n, M)
        order = np.argsort(L, axis=0)
        L_sorted = L[order, np.arange(M)[None, :]]
        gaps = L_sorted[1:, :] - L_sorted[:-1, :]
        overlaps = np.minimum(gaps, w[None, :])
        union_lengths = w + overlaps.sum(axis=0)
        return union_lengths.mean()

    # Initialize with the Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)

    # Projected subgradient descent with Nesterov acceleration
    best_x = x.copy()
    best_area = compute_area(x)

    # Adaptive parameters
    gamma = 0.005  # initial step size
    beta = 0.9     # momentum
    n_iters = 3000

    v = np.zeros(n)  # velocity for Nesterov
    x_nesterov = x.copy()

    for it in range(n_iters):
        # Compute subgradient at current x
        g = compute_grad(x)

        # Update velocity (Nesterov)
        v = beta * v + (1 - beta) * g
        x_new = x - gamma * v

        # Evaluate
        area_new = compute_area(x_new)

        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()

        # Adaptive step size: decrease if area increased, increase slowly if decreased
        if it > 0:
            if area_new > best_area * 1.001:
                gamma *= 0.95
            elif area_new < best_area * 0.999:
                gamma *= 1.01

        x = x_new

        # Decay step size slowly
        if it % 100 == 0 and it > 0:
            gamma *= 0.98

        # Print progress
        if it % 500 == 0:
            print(f"Iter {it}: area = {best_area:.10f}, gamma = {gamma:.6f}")

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
