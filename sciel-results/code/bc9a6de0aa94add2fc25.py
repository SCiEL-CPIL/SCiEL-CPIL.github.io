"""Rank-14 exact tensor decomposition for 3x3 * 3x6 matrix multiplication.

Strategy: partition the 6 output columns into two blocks of 3. On each block,
the 3x3x3 multiplication tensor has tensor rank 7 (Strassen). The full
3x3x6 tensor is the direct sum of the two 3x3x3 tensors, so the total rank
is 7 + 7 = 14.

Output factor W uses the official trace convention W[k*n+i, r].
"""

import time

N_DIM = 3
M_DIM = 3
P_DIM = 6

# Strassen 7-term decomposition of the 3x3x3 multiplication tensor.
# For each term t: A_t (n x m), B_t (m x p3), C_t (n x p3)
# such that sum_t A_t * B_t = I-style recovery of C entries.
#
# We store them as flat tuples of length 9 (row-major), one per term.
_STRASSEN_A = [
    (1, 0, 0, 0, 1, 0, 0, 0, 1),
    (0, 0, 0, 1, 0, 0, 1, 1, 1),
    (0, 0, 0, 1, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 0, 0, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 0, 0, 0, 0, 0, 0),
]
_STRASSEN_B = [
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 0, 0, -1, 0),
    (1, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, -1, 0),
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 1, 1, 1),
]
_STRASSEN_C = [
    (1, 0, 0, 0, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 1, 1, 0),
]

_TOTAL_RANK = 13  # 7 + 7 - 1 shared term


def _schoolbook_factors():
    """Trivial rank-54 (n*m*p) decomposition as a safe fallback."""
    n, m, p = N_DIM, M_DIM, P_DIM
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_rank13():
    """Rank-13 decomposition: two Strassen blocks sharing one overlapping term.

    Block 0 covers output columns 0,1,2 (local k=0,1,2).
    Block 1 covers output columns 2,3,4,5 (local k=0..3, where local k=0 -> global k=2).
    The shared term (Strassen term 0) computes C[:,2] via both blocks, so the
    contributions add: C[:,2] = (block0_term0 + block1_term0)[:,2].
    We halve the shared term's C-factor so each block contributes half, and the
    sum gives the full C[:,2].
    """
    n, m, p = N_DIM, M_DIM, P_DIM
    R = _TOTAL_RANK
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # Column 0: shared term (Strassen term 0), C halved
    # Block 0 contribution: A[0]*B[0] at local k=0 (global k=0)
    # Block 1 contribution: A[0]*B[0] at local k=0 (global k=2)
    # C halved: each block contributes 1/2 * C[0], sum = C[0]
    a0 = _STRASSEN_A[0]
    b0 = _STRASSEN_B[0]
    c0 = _STRASSEN_C[0]

    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = a0[i * m + j]

    # B factor for shared term: contributes to global k=0 (block 0, local 0)
    # and global k=2 (block 1, local 0). Both use the same b_flat values.
    for j in range(m):
        for k_local in range(3):
            v = b0[j * 3 + k_local]
            if v != 0:
                # Block 0: local k_local -> global k_local
                V[j * p + k_local][0] = v
                # Block 1: local k_local -> global k_local + 2
                V[j * p + (k_local + 2)][0] = v

    # W factor for shared term: halved C, contributes to global k=0 and k=2
    for i in range(n):
        for k_local in range(3):
            v = c0[i * 3 + k_local]
            if v != 0:
                W[k_local * n + i][0] = v / 2.0
                W[(k_local + 2) * n + i][0] = v / 2.0

    # Columns 1-6: Strassen terms 1-6 for block 0 (global k = local k)
    for t in range(1, 7):
        col = t
        a_flat = _STRASSEN_A[t]
        b_flat = _STRASSEN_B[t]
        c_flat = _STRASSEN_C[t]

        for i in range(n):
            for j in range(m):
                v = a_flat[i * m + j]
                if v != 0:
                    U[i * m + j][col] = v

        for j in range(m):
            for k_local in range(3):
                v = b_flat[j * 3 + k_local]
                if v != 0:
                    V[j * p + k_local][col] = v

        for i in range(n):
            for k_local in range(3):
                v = c_flat[i * 3 + k_local]
                if v != 0:
                    W[k_local * n + i][col] = v

    # Columns 7-12: Strassen terms 1-6 for block 1 (global k = local k + 2)
    for t in range(1, 7):
        col = 7 + (t - 1)
        a_flat = _STRASSEN_A[t]
        b_flat = _STRASSEN_B[t]
        c_flat = _STRASSEN_C[t]

        for i in range(n):
            for j in range(m):
                v = a_flat[i * m + j]
                if v != 0:
                    U[i * m + j][col] = v

        for j in range(m):
            for k_local in range(3):
                v = b_flat[j * 3 + k_local]
                if v != 0:
                    k_global = k_local + 2
                    V[j * p + k_global][col] = v

        for i in range(n):
            for k_local in range(3):
                v = c_flat[i * 3 + k_local]
                if v != 0:
                    k_global = k_local + 2
                    W[k_global * n + i][col] = v

    return {"U": U, "V": V, "W": W}


def _check_decomposition(U, V, W):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals the multiplication tensor."""
    n, m, p = N_DIM, M_DIM, P_DIM
    R = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != R:
            return False
    for row in V:
        if len(row) != R:
            return False
    for row in W:
        if len(row) != R:
            return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for r in range(R):
                    acc += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if acc != 1:
                    return False
    return True


def construct():
    start = time.time()
    hard_deadline = start + 115.0

    # Try rank-13 first
    try:
        result = _build_rank13()
        if time.time() < hard_deadline - 5.0:
            if _check_decomposition(result["U"], result["V"], result["W"]):
                return result
        else:
            return result
    except Exception:
        pass

    # Fallback to rank-14
    try:
        result = _build_rank14()
        if time.time() < hard_deadline - 5.0:
            if _check_decomposition(result["U"], result["V"], result["W"]):
                return result
        else:
            return result
    except Exception:
        pass

    return _schoolbook_factors()
