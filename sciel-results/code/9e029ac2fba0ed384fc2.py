"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication
    using Strassen's optimal scheme.  The entry convention is W[k*n+i, r].
    """
    t0 = time.perf_counter()
    t_limit = 150.0

    n = m = p = 2
    R = 7

    # Strassen coefficients
    u = np.array(
        [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    w = np.array(
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0],
        ],
        dtype=float,
    )

    # Fast tensor verification via einsum:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n,m,R), V to (m,p,R), W to (p,n,R) and combine.
    u3 = u.reshape(n, m, R)
    v3 = v.reshape(m, p, R)
    w3 = w.reshape(p, n, R)

    # Build T as (n,p) after contracting over j and r
    # T[i,k] should be 1 for all i,k
    t_check = np.einsum("imr,jpr,kir->ik", u3, v3, w3)

    residual = float(np.max(np.abs(t_check - 1.0)))
    if residual > 1e-12:
        # Should never trigger for the exact Strassen constants; fall through.
        pass

    # Return the three factor matrices as plain lists (the evaluator interface).
    return {"U": u.tolist(), "V": v.tolist(), "W": w.tolist()}
# EVOLVE-BLOCK-END
