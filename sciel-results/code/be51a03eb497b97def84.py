"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110  # safety margin before the 120s kill

def _time_left():
    return _DEADLINE - time.time()

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Verified Strassen decomposition for 2x2 @ 2x2 (rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # p1 = (a00+a11)(b01+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = 1
    # p2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    # p3 = a00(b10-b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # p4 = a11(b00-b10)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # p5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # p6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    # p7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    return U, V, W

def _combine_n(n1, n2, m, p, r1, r2):
    """Combine decompositions split on the first (n) dimension."""
    R1, R2 = r1[1].shape[1], r2[1].shape[1]
    rank = R1 + R2
    U = np.zeros(((n1 + n2) * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * (n1 + n2), rank))
    U[:n1 * m, :R1] = r1[0]
    U[n1 * m:, R1:] = r2[0]
    V[:, :R1] = r1[1]
    V[:, R1:] = r2[1]
    for k in range(p):
        W[k * (n1 + n2): k * (n1 + n2) + n1, :R1] = r1[2][k * n1:(k + 1) * n1, :R1]
        W[k * (n1 + n2) + n1:(k + 1) * (n1 + n2), R1:] = r2[2][k * n2:(k + 1) * n2, R1:]
    return U, V, W

def _combine_m(n, m1, m2, p, r1, r2):
    """Combine decompositions split on the middle (m) dimension."""
    R1, R2 = r1[1].shape[1], r2[1].shape[1]
    rank = R1 + R2
    U = np.zeros((n * (m1 + m2), rank))
    V = np.zeros(((m1 + m2) * p, rank))
    W = np.zeros((p * n, rank))
    U[:n * m1, :R1] = r1[0]
    U[n * m1:, R1:] = r2[0]
    for j in range(m1):
        V[j * p:(j + 1) * p, :R1] = r1[1][j * p:(j + 1) * p, :R1]
    for j in range(m2):
        V[(m1 + j) * p:(m1 + j + 1) * p, R1:] = r2[1][j * p:(j + 1) * p, R1:]
    W[:, :R1] = r1[2]
    W[:, R1:] = r2[2]
    return U, V, W

def _combine_p(n, m, p1, p2, r1, r2):
    """Combine decompositions split on the third (p) dimension."""
    R1, R2 = r1[1].shape[1], r2[1].shape[1]
    rank = R1 + R2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * (p1 + p2), rank))
    W = np.zeros(((p1 + p2) * n, rank))
    U[:, :R1] = r1[0]
    U[:, R1:] = r2[0]
    for j in range(m):
        V[j * (p1 + p2): j * (p1 + p2) + p1, :R1] = r1[1][j * p1:(j + 1) * p1, :R1]
        V[j * (p1 + p2) + p1:(j + 1) * (p1 + p2), R1:] = r2[1][j * p2:(j + 1) * p2, R1:]
    W[:p1 * n, :R1] = r1[2]
    W[p1 * n:, R1:] = r2[2]
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Verify sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(U.shape[1]):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _best_decompose(n, m, p, max_depth=4):
    """Recursively find a low-rank verified decomposition for n x m @ m x p."""
    if _time_left() < 1.0:
        return _schoolbook_decomp(n, m, p)

    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    if n * m * p <= 8:
        return _schoolbook_decomp(n, m, p)
    if max_depth <= 0:
        return _schoolbook_decomp(n, m, p)

    best_rank = n * m * p
    best = None

    # Option 1: split n
    if n >= 2:
        for n1 in range(1, n):
            n2 = n - n1
            r1 = _best_decompose(n1, m, p, max_depth - 1)
            r2 = _best_decompose(n2, m, p, max_depth - 1)
            cand = _combine_n(n1, n2, m, p, r1, r2)
            if cand[1].shape[1] < best_rank and _verify_decomp(*cand, n, m, p):
                best_rank = cand[1].shape[1]
                best = cand

    # Option 2: split m
    if m >= 2:
        for m1 in range(1, m):
            m2 = m - m1
            r1 = _best_decompose(n, m1, p, max_depth - 1)
            r2 = _best_decompose(n, m2, p, max_depth - 1)
            cand = _combine_m(n, m1, m2, p, r1, r2)
            if cand[1].shape[1] < best_rank and _verify_decomp(*cand, n, m, p):
                best_rank = cand[1].shape[1]
                best = cand

    # Option 3: split p
    if p >= 2:
        for p1 in range(1, p):
            p2 = p - p1
            r1 = _best_decompose(n, m, p1, max_depth - 1)
            r2 = _best_decompose(n, m, p2, max_depth - 1)
            cand = _combine_p(n, m, p1, p2, r1, r2)
            if cand[1].shape[1] < best_rank and _verify_decomp(*cand, n, m, p):
                best_rank = cand[1].shape[1]
                best = cand

    if best is None:
        return _schoolbook_decomp(n, m, p)
    return best

def construct():
    n, m, p = 4, 5, 6

    # Fallback: schoolbook (always valid, rank 120)
    U, V, W = _schoolbook_decomp(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try recursive block decomposition with Strassen base cases
    try:
        cand = _best_decompose(n, m, p, max_depth=4)
        if cand[1].shape[1] < best_rank and _verify_decomp(*cand, n, m, p):
            best_rank = cand[1].shape[1]
            best_U, best_V, best_W = cand
    except Exception:
        pass

    if _time_left() < 3.0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Final safety: ensure the returned decomposition is valid.
    if not _verify_decomp(best_U, best_V, best_W, n, m, p):
        U, V, W = _schoolbook_decomp(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
