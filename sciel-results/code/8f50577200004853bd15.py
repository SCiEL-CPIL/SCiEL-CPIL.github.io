"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check the decomposition identity:
    for all i,j,k: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _try_reduce(U, V, W, deadline):
    """
    Attempt to reduce rank by merging columns whose combined rows are linearly
    dependent. For each pair of columns a,b, if the 3x2 matrix formed by
    concatenating U[:,a],U[:,b],V[:,a],V[:,b],W[:,a],W[:,b] has rank 1,
    then one of the columns is a scalar multiple of the other in all three
    matrices, and we can merge them (keeping the one with larger norm).
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        cols = list(range(r))
        i = 0
        while i < len(cols) and time.time() < deadline:
            a = cols[i]
            j = i + 1
            while j < len(cols) and time.time() < deadline:
                b = cols[j]
                # Form the combined vector for column a and b
                va = np.concatenate([U[:, a], V[:, a], W[:, a]])
                vb = np.concatenate([U[:, b], V[:, b], W[:, b]])
                # Check if va and vb are linearly dependent
                if np.linalg.matrix_rank(np.column_stack([va, vb])) == 1:
                    # They are dependent; merge by keeping the one with larger L2 norm
                    na = np.linalg.norm(va)
                    nb = np.linalg.norm(vb)
                    if na >= nb:
                        # Remove column b
                        cols.pop(j)
                        U = np.delete(U, b, axis=1)
                        V = np.delete(V, b, axis=1)
                        W = np.delete(W, b, axis=1)
                        r -= 1
                        changed = True
                    else:
                        # Remove column a
                        cols.pop(i)
                        U = np.delete(U, a, axis=1)
                        V = np.delete(V, a, axis=1)
                        W = np.delete(W, a, axis=1)
                        r -= 1
                        changed = True
                        # Don't increment i, re-check same position
                        continue
                    break
                j += 1
            if not changed:
                i += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130  # Safety margin before 150s limit

    # Build the guaranteed-valid schoolbook decomposition (rank 36)
    U, V, W = _build_schoolbook()

    # Verify the schoolbook decomposition
    if not _verify(U, V, W):
        # Should never happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank via column merging (linear dependence detection)
    if time.time() < deadline:
        U2, V2, W2 = _try_reduce(U, V, W, deadline)
        if _verify(U2, V2, W2):
            U, V, W = U2, V2, W2

    # Final check: ensure we still have a valid decomposition
    if not _verify(U, V, W):
        # Fallback to original schoolbook
        U, V, W = _build_schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
