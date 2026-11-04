"""4x4x4 matrix multiplication tensor decomposition.

Strategy: build the provably-exact rank-49 Kronecker-of-Strassen decomposition,
then attempt exact redundant-rank elimination to reduce the rank while preserving
exactness. Falls back to the full rank-49 (or schoolbook rank-64) if needed.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Rank-7 Strassen decomposition of 2x2x2 matrix multiplication.

    Returns U, V, W as (4,7) arrays.
      U[i*2+j, r] = coefficient of A[i,j] in rank term r
      V[j*2+k, r] = coefficient of B[j,k] in rank term r
      W[k*2+i, r] = coefficient of C[i,k] in rank term r  (trace convention)

    Rank terms:
      P1 = (A00+A11)(B00+B11)
      P2 = (A10+A11) B00
      P3 = A00 (B01-B11)
      P4 = A11 (B10-B00)
      P5 = (A00+A01) B11
      P6 = (A10-A00)(B00+B01)
      P7 = (A01-A11)(B10+B11)

    C00 = P1 + P4 - P5 - P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P7
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1   # A00, A11
    V[0, 0] += 1; V[3, 0] += 1   # B00, B11
    W[0, 0] += 1; W[3, 0] += 1   # C00 (k=0,i=0), C11 (k=1,i=1)

    # P2 = (A10 + A11) B00
    U[2, 1] += 1; U[3, 1] += 1   # A10, A11
    V[0, 1] += 1                 # B00
    W[2, 1] += 1                 # C10 (k=1,i=0)

    # P3 = A00 (B01 - B11)
    U[0, 2] += 1                 # A00
    V[1, 2] += 1; V[3, 2] -= 1   # B01, -B11
    W[1, 2] += 1; W[3, 2] += 1   # C01 (k=0,i=1), C11 (k=1,i=1)

    # P4 = A11 (B10 - B00)
    U[3, 3] += 1                 # A11
    V[2, 3] += 1; V[0, 3] -= 1   # B10, -B00
    W[0, 3] += 1; W[2, 3] += 1   # C00 (k=0,i=0), C10 (k=1,i=0)

    # P5 = (A00 + A01) B11
    U[0, 4] += 1; U[1, 4] += 1   # A00, A01
    V[3, 4] += 1                 # B11
    W[0, 4] -= 1; W[1, 4] += 1   # -C00, C01

    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1   # A10, -A00
    V[0, 5] += 1; V[1, 5] += 1   # B00, B01
    W[2, 5] += 1                 # C10 (k=1,i=0)

    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1   # A01, -A11
    V[2, 6] += 1; V[3, 6] += 1   # B10, B11
    W[0, 6] -= 1; W[3, 6] += 1   # -C00, C11

    return U, V, W


def _build_mm_tensor(n, m, p):
    """Build the n x m x p matrix multiplication tensor T[i*m+j, j*p+k, k*n+i]=1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _decompose_tensor(U, V, W):
    """Return sum_r U[:,r] outer V[:,r] outer W[:,r]."""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _verify(U, V, W, T, tol=1e-9):
    """Check exact decomposition of T."""
    T_recon = _decompose_tensor(U, V, W)
    return np.max(np.abs(T_recon - T)) < tol


def _kron_decompose_4x4x4():
    """Rank-49 decomposition via Kronecker product of two rank-7 Strassen.

    U4[a, r*7+s] = U2[a2, r] * U2[a1, s], with a = a2*4 + a1.
    """
    U2, V2, W2 = _strassen_2x2()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)

    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            U4[:, col] = np.kron(U2[:, r], U2[:, s])
            V4[:, col] = np.kron(V2[:, r], V2[:, s])
            W4[:, col] = np.kron(W2[:, r], W2[:, s])

    return U4, V4, W4


def _greedy_reduce(U, V, W, T, tol=1e-9, deadline=None):
    """Try to remove redundant rank terms while preserving exactness.

    A rank term r is redundant if the tensor it contributes is a linear
    combination of the contributions of the remaining terms. We test this
    exactly: remove term r and check whether the remaining terms still
    reproduce T. We iterate to a fixed point.
    """
    U = U.copy(); V = V.copy(); W = W.copy()
    rank = U.shape[1]
    keep = list(range(rank))

    # Precompute per-term tensor contributions for fast re-summing.
    # term[r] = U[:,r] outer V[:,r] outer W[:,r]
    terms = np.stack([_decompose_tensor(U[:, [r]], V[:, [r]], W[:, [r]]) for r in range(rank)], axis=0)
    # terms shape (rank, 16, 16, 16)

    changed = True
    while changed:
        changed = False
        if deadline is not None and time.time() > deadline:
            break
        current_keep = list(keep)
        for r in current_keep:
            if deadline is not None and time.time() > deadline:
                break
            # candidate: drop r
            cand = [c for c in current_keep if c != r]
            if not cand:
                continue
            recon = terms[cand].sum(axis=0)
            if np.max(np.abs(recon - T)) < tol:
                keep = cand
                changed = True
                break  # restart scan with the reduced set

    idx = keep
    return U[:, idx], V[:, idx], W[:, idx]


def _schoolbook(n, m, p):
    """Rank n*m*p trivial decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 150s limit

    T = _build_mm_tensor(n, m, p)

    # Deterministic rank-49 Kronecker construction (exact, O(1) time).
    U, V, W = _kron_decompose_4x4x4()

    if not _verify(U, V, W, T):
        # Should never happen; fall back to schoolbook rank 64.
        U, V, W = _schoolbook(n, m, p)
    else:
        # Attempt exact redundant-rank elimination (only helps, never hurts).
        U_red, V_red, W_red = _greedy_reduce(U, V, W, T, deadline=deadline)
        if _verify(U_red, V_red, W_red, T):
            U, V, W = U_red, V_red, W_red

    # Convert to list of lists (integer/half-integer entries preserved as floats).
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
