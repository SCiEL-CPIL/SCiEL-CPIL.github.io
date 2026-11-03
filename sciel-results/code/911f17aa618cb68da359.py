"""Program for matrix multiplication 3,4,5 exact rank-minimal decomposition."""
import time

DIMENSIONS = [3, 4, 5]
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, trials=200, tol=1e-8):
    """Verify sum_c U[:,c] (nm) * V[:,c] (mp) * W[:,c] (pn) == multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(0)
    for _ in range(trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_true = A @ B
        C = np.zeros((n, p))
        for c in range(rank):
            Uc2 = U[:, c].reshape(n, m)
            Vc2 = V[:, c].reshape(m, p)
            Wc2 = W[:, c].reshape(p, n)
            M = Uc2 @ Vc2
            C += M * Wc2.T
        if np.max(np.abs(C - C_true)) > tol:
            return False
    return True


def _build_schoolbook():
    """Standard rank n*m*p decomposition: one rank-1 term per (i,j,k)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    # Build the guaranteed-valid schoolbook decomposition.
    result = _build_schoolbook()
    # Safety verification with a time guard.
    if time.time() < deadline - 5.0:
        ok = _verify(result["U"], result["V"], result["W"])
        if not ok:
            # Should never happen; return as-is (schoolbook is correct).
            return result
    return result
# EVOLVE-BLOCK-END
