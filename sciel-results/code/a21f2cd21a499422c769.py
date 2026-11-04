"""Program that searches for a low-rank exact tensor decomposition of 2x3x3 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N, M, P = n * m, m * p, p * n
SCHOOOOK_RANK = n * m * p  # 18


def _verify(U, V, W, tol=1e-9):
    """Check U * outer(V, W) matches the tensor exactly."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Build tensor T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # Use einsum for efficiency
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    # Expected tensor: T[i,j,k] = 1 iff i<m, j<m, k<p (for A[i,j]*B[j,k] in C[i,k])
    # With the trace convention, the tensor is T[i,j,k] = delta_{j in 0..m-1}
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p, else 0
    # But we need the right indexing. Let's build it explicitly.
    expected = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0
    if T.shape != expected.shape:
        return False
    if np.max(np.abs(T - expected)) > tol:
        return False
    return True


def _try_random_decomposition(rng, max_iters=500000, time_limit=10.0):
    """Try random rank-16 decompositions with verification."""
    import time
    t0 = time.time()
    rank = 16  # Try rank 16 (known lower bound for 2x3x3 is 16)
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        # Initialize with small random values
        U = rng.uniform(-1, 1, size=(N, rank))
        V = rng.uniform(-1, 1, size=(M, rank))
        W = rng.uniform(-1, 1, size=(P, rank))
        if _verify(U, V, W):
            return rank, U, V, W
    return None


def _try_gradient_descent(rng, time_limit=30.0):
    """Use gradient descent to minimize ||U*outer(V,W) - T||^2 for rank 16."""
    import time
    t0 = time.time()
    rank = 16
    expected = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0

    best_loss = np.inf
    best_U, best_V, best_W = None, None, None

    for restart in range(20):
        if time.time() - t0 > time_limit:
            break
        U = rng.normal(0, 0.5, size=(N, rank))
        V = rng.normal(0, 0.5, size=(M, rank))
        W = rng.normal(0, 0.5, size=(P, rank))

        lr = 0.01
        for step in range(5000):
            if time.time() - t0 > time_limit:
                break
            # Compute loss and gradients
            T = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = T - expected
            loss = np.sum(diff ** 2)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-12:
                return rank, U, V, W

            # Gradients
            dU = np.einsum('ijk,jc,kc->ic', diff, V, W) * 2.0
            dV = np.einsum('ijk,ic,kc->jc', diff, U, W) * 2.0
            dW = np.einsum('ijk,ic,jc->kc', diff, U, V) * 2.0

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            # Adaptive learning rate
            if loss > best_loss * 1.5:
                lr *= 0.5
            elif step % 100 == 0 and step > 0:
                lr = min(lr * 1.05, 0.1)

    if best_U is not None and best_loss < 1e-9:
        return rank, best_U, best_V, best_W
    return None


def _try_integer_search(rng, time_limit=20.0):
    """Try structured integer/half-integer rank-16 decompositions."""
    import time
    t0 = time.time()
    rank = 16

    # For 2x3x3, the tensor has 18 ones. We need 16 rank-1 terms.
    # This means some rank-1 terms must contribute to multiple positions.
    # A rank-1 term U[:,c] * V[:,c] * W[:,c] contributes to positions (i,j,k)
    # where all three are nonzero.

    # Strategy: try to find a structured decomposition.
    # For 2x3x3, one known approach:
    # The tensor can be seen as block diagonal in some sense.
    # A[0,:] * B[:,k] for i=0, and A[1,:] * B[:,k] for i=1
    # Each row of A times B: 3x3 matrix mult takes rank 9 schoolbook.
    # Two rows: 18 total. Need to save 2.

    # Let me try a different approach: use the fact that for 2x3x3,
    # we can potentially share some computation between the two rows.

    # Actually, let me just try random integer/half-integer searches.
    values = [0, 0.5, 1, -0.5, -1]
    n_vals = len(values)

    for it in range(50000):
        if time.time() - t0 > time_limit:
            break
        # Randomly pick a few nonzero entries
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))

        for c in range(rank):
            # Each column has some sparsity
            nz_u = rng.integers(1, 4)  # 1-3 nonzeros
            nz_v = rng.integers(1, 4)
            nz_w = rng.integers(1, 4)

            idx_u = rng.choice(N, size=nz_u, replace=False)
            idx_v = rng.choice(M, size=nz_v, replace=False)
            idx_w = rng.choice(P, size=nz_w, replace=False)

            for i in idx_u:
                U[i, c] = rng.choice(values)
            for j in idx_v:
                V[j, c] = rng.choice(values)
            for k in idx_w:
                W[k, c] = rng.choice(values)

        if _verify(U, V, W):
            return rank, U, V, W

    return None


# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    deadline = 130.0  # Safety margin before 150s

    # Default: schoolbook rank-18
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rng = np.random.default_rng(42)

    # Try gradient descent for rank 16
    if time.time() - t0 < deadline - 35:
        result = _try_gradient_descent(rng, time_limit=30.0)
        if result is not None:
            rank, U, V, W = result
            if _verify(U, V, W):
                U_list = [[int(round(x * 2)) / 2 for x in row] for row in U]
                V_list = [[int(round(x * 2)) / 2 for x in row] for row in V]
                W_list = [[int(round(x * 2)) / 2 for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Try integer search
    if time.time() - t0 < deadline - 25:
        result = _try_integer_search(rng, time_limit=20.0)
        if result is not None:
            rank, U, V, W = result
            if _verify(U, V, W):
                U_list = [[int(round(x * 2)) / 2 for x in row] for row in U]
                V_list = [[int(round(x * 2)) / 2 for x in row] for row in V]
                W_list = [[int(round(x * 2)) / 2 for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Try random search
    if time.time() - t0 < deadline - 15:
        result = _try_random_decomposition(rng, max_iters=200000, time_limit=12.0)
        if result is not None:
            rank, U, V, W = result
            if _verify(U, V, W):
                U_list = [[int(round(x * 2)) / 2 for x in row] for row in U]
                V_list = [[int(round(x * 2)) / 2 for x in row] for row in V]
                W_list = [[int(round(x * 2)) / 2 for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
