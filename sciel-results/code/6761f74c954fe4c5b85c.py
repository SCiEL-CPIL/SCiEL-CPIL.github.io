"""Matrix multiplication 4x5x6 tensor decomposition - rank minimization."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_222():
    """Strassen decomposition for 2x2x2 with rank 7."""
    rank = 7
    U = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
    ], dtype=float)
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _combine_split_m(n, m1, m2, p):
    """
    Split m into m1+m2. C = A[:,:m1]*B[:m1,:] + A[:,m1:]*B[m1:,:]
    Returns decomposition for n x (m1+m2) x p.
    """
    # Get sub-decompositions
    Ua, Va, Wa = _solve(n, m1, p)
    Ub, Vb, Wb = _solve(n, m2, p)
    ra = Ua.shape[1]
    rb = Ub.shape[1]
    rank = ra + rb
    m = m1 + m2

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for i in range(n):
        for j in range(m1):
            U[i * m + j, :ra] = Ua[i * m1 + j, :]
        for j in range(m1, m):
            U[i * m + j, ra:] = Ub[i * m2 + (j - m1), :]

    for j in range(m1):
        for k in range(p):
            V[j * p + k, :ra] = Va[j * p + k, :]
    for j in range(m1, m):
        for k in range(p):
            V[j * p + k, ra:] = Vb[(j - m1) * p + k, :]

    for k in range(p):
        for i in range(n):
            W[k * n + i, :ra] = Wa[k * n + i, :]
            W[k * n + i, ra:] = Wb[k * n + i, :]

    return U, V, W


def _combine_split_n(n1, n2, m, p):
    """
    Split n into n1+n2. C_top = A_top*B, C_bottom = A_bottom*B.
    Both share the same B, so V must be combined.
    """
    Ua, Va, Wa = _solve(n1, m, p)
    Ub, Vb, Wb = _solve(n2, m, p)
    ra = Ua.shape[1]
    rb = Ub.shape[1]
    rank = ra + rb
    n = n1 + n2

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for i in range(n1):
        for j in range(m):
            U[i * m + j, :ra] = Ua[i * m + j, :]
    for i in range(n1, n):
        for j in range(m):
            U[i * m + j, ra:] = Ub[(i - n1) * m + j, :]

    # V: both sub-problems use the same B entries
    # We need V such that for the first ra columns, it matches Va, and for last rb, matches Vb
    # But this doesn't work directly because both use all of B.
    # The correct approach: V[j*p+k, :ra] = Va[j*p+k, :] and V[j*p+k, ra:] = Vb[j*p+k, :]
    # This works because the decomposition sums over c, and the contributions from
    # the two blocks are separated by the column index.
    for j in range(m):
        for k in range(p):
            V[j * p + k, :ra] = Va[j * p + k, :]
            V[j * p + k, ra:] = Vb[j * p + k, :]

    for k in range(p):
        for i in range(n1):
            W[k * n + i, :ra] = Wa[k * n1 + i, :]
        for i in range(n1, n):
            W[k * n + i, ra:] = Wb[k * n2 + (i - n1), :]

    return U, V, W


def _combine_split_p(n, m, p1, p2):
    """
    Split p into p1+p2. C_left = A*B_left, C_right = A*B_right.
    Both share the same A, so U must be combined.
    """
    Ua, Va, Wa = _solve(n, m, p1)
    Ub, Vb, Wb = _solve(n, m, p2)
    ra = Ua.shape[1]
    rb = Ub.shape[1]
    rank = ra + rb
    p = p1 + p2

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # U: both sub-problems use the same A entries
    for i in range(n):
        for j in range(m):
            U[i * m + j, :ra] = Ua[i * m + j, :]
            U[i * m + j, ra:] = Ub[i * m + j, :]

    for j in range(m):
        for k in range(p1):
            V[j * p + k, :ra] = Va[j * p1 + k, :]
        for k in range(p1, p):
            V[j * p + k, ra:] = Vb[j * p2 + (k - p1), :]

    for k in range(p1):
        for i in range(n):
            W[k * n + i, :ra] = Wa[k * n + i, :]
    for k in range(p1, p):
        for i in range(n):
            W[k * n + i, ra:] = Wb[(k - p1) * n + i, :]

    return U, V, W


def _solve(n, m, p, deadline=None):
    """
    Recursive solver for tensor decomposition of n x m x p matrix multiplication.
    Returns (U, V, W) as numpy arrays.
    """
    if deadline is not None and time.time() > deadline:
        return _schoolbook(n, m, p)

    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()

    if n == 1 and m == 1 and p == 1:
        U = np.ones((1, 1))
        V = np.ones((1, 1))
        W = np.ones((1, 1))
        return U, V, W

    if m == 1:
        return _schoolbook(n, 1, p)
    if n == 1:
        return _schoolbook(1, m, p)
    if p == 1:
        return _schoolbook(n, m, 1)

    # Try all splits and pick the one with minimum rank
    best_rank = float('inf')
    best_result = None

    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U, V, W = _combine_split_m(n, m1, m2, p)
        r = U.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = (U, V, W)
        if deadline is not None and time.time() > deadline:
            break

    if deadline is not None and time.time() > deadline:
        if best_result is not None:
            return best_result
        return _schoolbook(n, m, p)

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U, V, W = _combine_split_n(n1, n2, m, p)
        r = U.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = (U, V, W)
        if deadline is not None and time.time() > deadline:
            break

    if deadline is not None and time.time() > deadline:
        if best_result is not None:
            return best_result
        return _schoolbook(n, m, p)

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U, V, W = _combine_split_p(n, m, p1, p2)
        r = U.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = (U, V, W)
        if deadline is not None and time.time() > deadline:
            break

    if best_result is not None:
        return best_result

    return _schoolbook(n, m, p)


def _rank_of(U, V, W):
    return U.shape[1]


def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x5 x 5x6 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W}
    """
    start = time.time()
    deadline = start + 140  # Leave 10s margin

    n, m, p = DIMENSIONS

    # First, get a baseline via recursive splitting
    try:
        U, V, W = _solve(n, m, p, deadline=deadline)
        rank = U.shape[1]
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
            rank = U.shape[1]
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        rank = U.shape[1]

    # Try to improve with random search if time allows
    if time.time() < deadline - 30:
        try:
            best_U, best_V, best_W = U, V, W
            best_rank = rank

            # Try various structured decompositions
            # Attempt: use Strassen-like patterns on sub-blocks

            # For 4x5x6, try decomposing as:
            # Split 4 into 2+2, 5 into 2+3, 6 into 3+3
            # Or other combinations

            # Try a direct search with small rank
            # The theoretical lower bound is max(n*m, m*p, p*n) = max(20, 30, 24) = 30
            # Schoolbook is 120. Recursive gives much less.
            # Let's try to see if we can do better than the recursive result.

            # Try different split strategies explicitly
            strategies = [
                # (type, params)
                ('m', 1, 4), ('m', 2, 3),
                ('n', 1, 3), ('n', 2, 2),
                ('p', 1, 5), ('p', 2, 4), ('p', 3, 3),
            ]

            for stype, s1, s2 in strategies:
                if time.time() > deadline - 10:
                    break
                try:
                    if stype == 'm':
                        U2, V2, W2 = _combine_split_m(n, s1, s2, p)
                    elif stype == 'n':
                        U2, V2, W2 = _combine_split_n(s1, s2, m, p)
                    else:
                        U2, V2, W2 = _combine_split_p(n, m, s1, s2)

                    r2 = U2.shape[1]
                    if r2 < best_rank and _verify(U2, V2, W2, n, m, p):
                        best_U, best_V, best_W = U2, V2, W2
                        best_rank = r2
                except Exception:
                    pass

            U, V, W = best_U, best_V, best_W
            rank = best_rank

        except Exception:
            pass

    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
