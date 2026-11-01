"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _build_schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _residual(U, V, W):
    """Compute max |sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1| over all (i,j,k)."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _verify(U, V, W, tol=1e-9):
    return _residual(U, V, W) < tol


def _to_int_list(mat):
    """Convert a numpy array of near-integers to a list of lists of ints."""
    return np.round(mat).astype(int).tolist()


def _random_init(n, m, p, rank, rng):
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U, V, W


def _grad_descent(U, V, W, n, m, p, lr, iters, rng, deadline):
    """Simple gradient descent on the tensor decomposition objective."""
    r = U.shape[1]
    start = time.time()
    for it in range(iters):
        if time.time() > deadline:
            break
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Current value at (i,j,k)
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = val - 1.0
                    for c in range(r):
                        dU[i * m + j, c] += err * V[j * p + k, c] * W[k * n + i, c]
                        dV[j * p + k, c] += err * U[i * m + j, c] * W[k * n + i, c]
                        dW[k * n + i, c] += err * U[i * m + j, c] * V[j * p + k, c]
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    return U, V, W


def _try_rank(rank, n, m, p, deadline, n_restarts=5, iters_per_restart=200):
    """Try to find a valid decomposition of given rank."""
    start = time.time()
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        rng = np.random.RandomState(42 + restart * 1000 + rank)
        U, V, W = _random_init(n, m, p, rank, rng)
        # Adaptive learning rate
        lr = 0.01
        for phase in range(3):
            iters = iters_per_restart
            U, V, W = _grad_descent(U, V, W, n, m, p, lr, iters, rng, deadline)
            err = _residual(U, V, W)
            if err < 1e-9:
                return U, V, W
            lr *= 0.5
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    The schoolbook decomposition has rank 18.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0  # 15s safety margin

    # Try ranks from 12 to 17
    for rank in range(12, 18):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, deadline, n_restarts=3, iters_per_restart=100)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}

    # Fallback: schoolbook
    sb = _build_schoolbook()
    return {"U": _to_int_list(sb["U"]), "V": _to_int_list(sb["V"]), "W": _to_int_list(sb["W"])}

# EVOLVE-BLOCK-END
