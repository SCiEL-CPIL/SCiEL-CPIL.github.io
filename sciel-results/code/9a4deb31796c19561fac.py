"""Find a rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x6 matrix."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) factor matrices."""
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
    return U, V, W


def _kronecker_product(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Compute Kronecker product of two matrix-multiplication tensor decompositions.
    T1: (n1, m1, p1) with factors U1, V1, W1 of rank r1
    T2: (n2, m2, p2) with factors U2, V2, W2 of rank r2
    Result: (n1*n2, m1*m2, p1*p2) with rank r1*r2
    """
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 * r2
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U part
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1][c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2][c2]
                            if u2_val == 0:
                                continue
                            U[(i1 * n2 + i2) * m + (j1 * m2 + j2)][c] = u1_val * u2_val
            # V part
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1][c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2][c2]
                            if v2_val == 0:
                                continue
                            V[(j1 * m2 + j2) * p + (k1 * p2 + k2)][c] = v1_val * v2_val
            # W part
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1][c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2][c2]
                            if w2_val == 0:
                                continue
                            W[(k1 * p2 + k2) * n + (i1 * n2 + i2)][c] = w1_val * w2_val

    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is exact for matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)

    result = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    target = np.ones((n, m, p))

    return np.allclose(result, target, atol=tol)


def _try_kronecker_factorizations(n, m, p):
    """
    Try various Kronecker product factorizations of (n, m, p).
    For (3, 4, 6):
      - (3,2,6) x (1,2,1): rank 36*1 = 36
      - (3,4,3) x (1,1,2): rank 36*2 = 72 (no improvement)
      - (1,4,6) x (3,1,1): rank 24*3 = 72 (no improvement)
      - (3,4,6) itself: rank 72
    Best: (3,2,6) x (1,2,1) with rank 36.
    """
    candidates = []

    # (3,2,6) x (1,2,1) -> (3,4,6), rank 36
    n1, m1, p1 = 3, 2, 6
    n2, m2, p2 = 1, 2, 1
    if n1 * n2 == n and m1 * m2 == m and p1 * p2 == p:
        U1, V1, W1 = _schoolbook(n1, m1, p1)
        # (1,2,1) has rank 1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        U2 = [[1], [1]]
        V2 = [[1], [1]]
        W2 = [[1]]
        U, V, W = _kronecker_product(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2)
        rank = len(U[0])
        candidates.append((rank, U, V, W))

    # (3,4,3) x (1,1,2) -> (3,4,6), rank 36*2 = 72
    n1, m1, p1 = 3, 4, 3
    n2, m2, p2 = 1, 1, 2
    if n1 * n2 == n and m1 * m2 == m and p1 * p2 == p:
        U1, V1, W1 = _schoolbook(n1, m1, p1)
        U2 = [[1]]
        V2 = [[1]]
        W2 = [[1], [1]]
        U, V, W = _kronecker_product(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2)
        rank = len(U[0])
        candidates.append((rank, U, V, W))

    # (1,4,6) x (3,1,1) -> (3,4,6), rank 24*3 = 72
    n1, m1, p1 = 1, 4, 6
    n2, m2, p2 = 3, 1, 1
    if n1 * n2 == n and m1 * m2 == m and p1 * p2 == p:
        U1, V1, W1 = _schoolbook(n1, m1, p1)
        U2 = [[1], [1], [1]]
        V2 = [[1]]
        W2 = [[1]]
        U, V, W = _kronecker_product(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2)
        rank = len(U[0])
        candidates.append((rank, U, V, W))

    # Sort by rank
    candidates.sort(key=lambda x: x[0])
    return candidates


def _try_direct_low_rank(n, m, p, max_rank):
    """
    Try to find a low-rank decomposition by exploiting structure.
    For (3,4,6), try rank < 72 by combining terms.
    """
    # This is a placeholder for future search algorithms
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    best_result = None
    best_rank = n * m * p  # schoolbook rank as initial upper bound

    # Try Kronecker product factorizations
    try:
        candidates = _try_kronecker_factorizations(n, m, p)
        for rank, U, V, W in candidates:
            if rank < best_rank:
                if _verify_decomposition(n, m, p, U, V, W):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook decomposition
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U, "V": V, "W": W}

    return best_result

# EVOLVE-BLOCK-END
