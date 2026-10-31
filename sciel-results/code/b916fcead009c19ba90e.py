"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def construct():
    # Build the autoconvolution constraint matrix A for l2 = 0
    # a_l2(k) = sum_{i=0}^{N-1-k} f[i] * f[i+k]
    # For each k in 0..N-1, the row of A has 1s at positions (i, i+k)
    A = np.zeros((N, N * N))
    for k in range(N):
        for i in range(N - k):
            A[k, i * N + (i + k)] = 1.0

    # Objective: minimize sum f[i] (since we can scale later, we want small sum
    # with a_k <= 1 and -a_k <= 1 for all k)
    c = np.ones(N * N)

    # Constraints: -1 <= a_k <= 1 for all k
    # a_k <= 1:  A @ x <= 1
    # -a_k <= 1: -A @ x <= 1
    A_ub = np.vstack([A, -A])
    b_ub = np.ones(2 * N)

    # Bounds: each f[i] in [-1, 1]
    bounds = [(-1.0, 1.0)] * (N * N)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success and res.x is not None:
        f = res.x.reshape(N, N)
        # Take the diagonal as the step heights
        f_step = np.diag(f)
    else:
        # Fallback to constant function
        f_step = np.ones(N)

    # Normalize so that sum f = 1 (to make the ratio scale-invariant and well-defined)
    s = np.sum(f_step)
    if abs(s) > 1e-12:
        f_step = f_step / s
    else:
        f_step = np.ones(N) / N

    return f_step
# EVOLVE-BLOCK-END
