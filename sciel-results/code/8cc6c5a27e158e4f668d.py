import numpy as np
import time

DIMENSIONS = [3, 3, 4]


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    return U, V, W


def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # T[i,j,k] should be 1
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Build via einsum
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    target = np.ones((n, m, p))
    return np.max(np.abs(T - target)) < tol


def _try_prune(U, V, W, col, tol=1e-6):
    """
    Try to remove column `col` by re-solving for U, V, W using the remaining columns.
    Returns (U_new, V_new, W_new, success) where shapes drop one column.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]

    # Remove the column
    mask = [r for r in range(rank) if r != col]
    U0 = U[:, mask]
    V0 = V[:, mask]
    W0 = W[:, mask]

    # We want to find U, V, W (each with `len(mask)` columns) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    #
    # This is a trilinear system. We can solve iteratively:
    # Fix V and W, solve for U (linear in U).
    # Fix U and W, solve for V.
    # Fix U and V, solve for W.
    # Iterate.

    nrm = len(mask)
    Uv = U0.copy()
    Vv = V0.copy()
    Wv = W0.copy()

    for it in range(50):
        # Solve for U: for each (i,j), U[i*m+j, r] * (V[j*p+? ,r]*W[?,r]) ...
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For fixed (i,j), we have p equations (k=0..p-1) in nrm unknowns (r).
        # A[i,j] is p x nrm, b[i,j] is p x 1
        # A[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
        # b[i,j,k] = 1
        # Solve least squares.
        U_new = np.zeros((n * m, nrm))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, nrm))
                for k in range(p):
                    A[k, :] = Vv[j * p + k, :] * Wv[k * n + i, :]
                b = np.ones(p)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i * m + j, :] = sol

        # Solve for V: for each (j,k), V[j*p+k, r] * (U[?,r]*W[k*n+?,r])
        # T[i,j,k] = sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r]
        # For fixed (j,k), n equations (i=0..n-1) in nrm unknowns.
        # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
        # b[i] = 1
        V_new = np.zeros((m * p, nrm))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, nrm))
                for i in range(n):
                    A[i, :] = U_new[i * m + j, :] * Wv[k * n + i, :]
                b = np.ones(n)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j * p + k, :] = sol

        # Solve for W: for each (k,i), W[k*n+i, r] * (U[?,r]*V[?,r])
        # T[i,j,k] = sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r]
        # For fixed (k,i), m equations (j=0..m-1) in nrm unknowns.
        # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
        # b[j] = 1
        W_new = np.zeros((p * n, nrm))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, nrm))
                for j in range(m):
                    A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                b = np.ones(m)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k * n + i, :] = sol

        Uv, Vv, Wv = U_new, V_new, W_new

    if _verify(Uv, Vv, Wv, tol):
        return Uv, Vv, Wv, True
    return None, None, None, False


def _prune_all(U, V, W, tol=1e-6, max_rounds=20):
    """
    Iteratively try to remove columns. Each round, try each column once.
    """
    n, m, p = DIMENSIONS
    for round_i in range(max_rounds):
        progress = False
        rank = U.shape[1]
        for col in range(rank - 1, -1, -1):
            U_new, V_new, W_new, ok = _try_prune(U, V, W, col, tol)
            if ok:
                U, V, W = U_new, V_new, W_new
                progress = True
        if not progress:
            break
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 135.0  # safety margin before 150s

    U, V, W = _schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]

    # Iterative pruning
    while True:
        elapsed = time.time() - start
        if elapsed > deadline:
            break
        U, V, W = _prune_all(U, V, W, tol=1e-6, max_rounds=5)
        rank = U.shape[1]
        if rank < best_rank:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
        # If no progress, break
        U2, V2, W2 = _prune_all(U, V, W, tol=1e-6, max_rounds=1)
        if U2.shape[1] == U.shape[1]:
            break
        U, V, W = U2, V2, W2

    # Final polish: try to reduce tolerance and re-verify
    U, V, W = best_U, best_V, best_W

    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    print("Rank:", U.shape[1])
    print("Valid:", _verify(U, V, W))
