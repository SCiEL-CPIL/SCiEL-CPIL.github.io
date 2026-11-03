"""Matrix multiplication 4x4x4 tensor decomposition.

Combines the verified Kronecker-of-Strassen construction (rank 49) with a
greedy column-removal refinement to strip linearly-redundant tensor columns.
The output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np
import time

# Safety margin before the 150s kill.
DEADLINE = 120.0


def _naive_fallback():
    """Schoolbook rank-64 decomposition; always valid."""
    U = np.zeros((16, 64), dtype=int)
    V = np.zeros((16, 64), dtype=int)
    W = np.zeros((16, 64), dtype=int)
    col = 0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                U[ii * 4 + jj, col] = 1
                V[jj * 4 + kk, col] = 1
                W[kk * 4 + ii, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 matmul.

    Index convention: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i.
    """
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00+A11)(B00+B11): C00 += M0, C11 += M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10+A11)B00: C10 += M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1

    # M2 = A00(B10-B11): C01 += M2
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1

    # M3 = A11(B01-B00): C01 -= M3, C11 += M3
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = 1

    # M4 = (A00+A01)B11: C00 -= M4, C01 += M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # M5 = (A10-A00)(B00+B01): C10 -= M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = -1

    # M6 = (A01-A11)(B10+B11): C10 += M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1

    # Verify 2x2 Strassen exactly.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(7):
                    actual += U2[i * 2 + j, c] * V2[j * 2 + k, c] * W2[k * 2 + i, c]
                if actual != expected:
                    return None
    return U2, V2, W2


def _perm_u(S):
    i = S // 4
    j = S % 4
    i1, i2 = i // 2, i % 2
    j1, j2 = j // 2, j % 2
    return (i1 * 2 + j1) * 4 + (i2 * 2 + j2)


def _perm_v(S):
    j = S // 4
    k = S % 4
    j1, j2 = j // 2, j % 2
    k1, k2 = k // 2, k % 2
    return (j1 * 2 + k1) * 4 + (j2 * 2 + k2)


def _perm_w(S):
    k = S // 4
    i = S % 4
    k1, k2 = k // 2, k % 2
    i1, i2 = i // 2, i % 2
    return (k1 * 2 + i1) * 4 + (k2 * 2 + i2)


def _verify(U, V, W, rank):
    """Check T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta(i,k)."""
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(rank):
                    actual += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                if actual != expected:
                    return False
    return True


def _greedy_reduce(U, V, W, start_time):
    """Greedy set-cover style removal of redundant tensor columns.

    Each column c defines a 64-dim contribution vector
    vec_c[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c].
    We need sum of active vec_c == target (delta(i,k)).
    A column is removable iff its vec is in the span of the remaining active
    columns AND removing it keeps the exact integer sum == target.
    """
    ncol = U.shape[1]
    # Target vector over (i,j,k) in order i major, j, k.
    target = np.zeros(64, dtype=int)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i == k:
                    target[idx] = 1
                idx += 1

    # Contribution matrix: rows = columns, cols = 64 entries.
    contrib = np.zeros((ncol, 64), dtype=int)
    for c in range(ncol):
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    contrib[c, idx] = U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                    idx += 1

    # Current residual: target - sum(active). Start active = all columns.
    active = list(range(ncol))
    residual = target - contrib.sum(axis=0)
    if not np.all(residual == 0):
        return None  # base decomposition is broken

    # Heuristic removal order: prefer columns with smaller L1 norm first
    # (they are "smaller" and more likely redundant), tie-break by index.
    norms = np.abs(contrib).sum(axis=1)
    order = sorted(range(ncol), key=lambda c: (norms[c], c))

    # For speed, maintain the span of the *remaining* active columns via
    # rank of the matrix restricted to active columns. But exact integer span
    # is costly; instead we use the exact-sum check which is necessary, and
    # use a rational/float rank test as a cheap sufficient filter.
    # A column c is removable iff:
    #   (a) residual_after = residual + contrib[c] == 0  (exact sum preserved)
    #   (b) contrib[c] is in the linear span of the other active columns
    #       (so the decomposition still "covers" the tensor structure).
    # Condition (a) alone already guarantees the final tensor equation holds,
    # because the sum of remaining active columns will equal target exactly.
    # So we only need (a): try removing c if residual + contrib[c] == 0.

    for c in order:
        if time.time() - start_time > DEADLINE - 5:
            break
        if c not in active:
            continue
        # Try removing c.
        new_residual = residual + contrib[c]
        if np.all(new_residual == 0):
            # Removing c keeps the exact sum. Remove it.
            active.remove(c)
            residual = new_residual

    if len(active) >= ncol:
        return None
    # Build reduced matrices.
    U_r = U[:, active]
    V_r = V[:, active]
    W_r = W[:, active]
    return U_r, V_r, W_r


def construct():
    """
    Constructs a low-rank exact tensor decomposition for 4x4 matrix
    multiplication. Starts from the verified Kronecker-of-Strassen rank-49
    decomposition and greedily removes redundant columns. Falls back to
    rank 64 (naive) if anything fails.
    """
    start_time = time.time()

    # 1) Build the verified rank-49 Kronecker-of-Strassen decomposition.
    s = _strassen_2x2()
    if s is None:
        return _naive_fallback()
    U2, V2, W2 = s

    U4_kron = np.kron(U2, U2)  # (16, 49)
    V4_kron = np.kron(V2, V2)  # (16, 49)
    W4_kron = np.kron(W2, W2)  # (16, 49)

    perm_U = [_perm_u(S) for S in range(16)]
    perm_V = [_perm_v(S) for S in range(16)]
    perm_W = [_perm_w(S) for S in range(16)]

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # Verify the rank-49 decomposition exactly.
    if not _verify(U4, V4, W4, 49):
        return _naive_fallback()

    # Keep rank 49 as the guaranteed-best-so-far.
    best = (U4, V4, W4)

    # 2) Attempt greedy column removal to reduce rank.
    reduced = _greedy_reduce(U4, V4, W4, start_time)
    if reduced is not None:
        Ur, Vr, Wr = reduced
        # Verify the reduced decomposition exactly before accepting.
        if _verify(Ur, Vr, Wr, Ur.shape[1]):
            if Ur.shape[1] < best[0].shape[1]:
                best = (Ur, Vr, Wr)

    # 3) Deadline guard: return best result found.
    if time.time() - start_time > DEADLINE - 2:
        pass  # already have best; just return it

    U, V, W = best
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
