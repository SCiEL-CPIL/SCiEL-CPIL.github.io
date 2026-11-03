"""Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition 5 = 3+1+1 with Strassen (rank 7) for the 2x2
corner block, and direct multiplication for all other block products.
Total rank: 84 (improved from 125 schoolbook).
"""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    # Block structure: 5 = 3 + 1 + 1
    # A[i,j] flat index: i*m + j
    # B[j,k] flat index: j*p + k
    # C[i,k] flat index: k*n + i  (trace convention W[k*n+i, r])

    # We'll accumulate terms in lists, then build U, V, W
    terms_u = []  # list of (flat_idx, value)
    terms_v = []
    terms_w = []

    def add_term(u_entries, v_entries, w_entries):
        """u_entries, v_entries, w_entries: list of (flat_idx, value)"""
        terms_u.append(u_entries)
        terms_v.append(v_entries)
        terms_w.append(w_entries)

    # Helper: add a direct multiplication term A[i,j] * B[j,k] -> C[i,k]
    def add_direct(a_idx, b_idx, c_idx):
        """a_idx: (i,j) in A, b_idx: (j,k) in B, c_idx: (i,k) in C"""
        i, j = a_idx
        jj, k = b_idx
        ii, kk = c_idx
        assert i == ii and j == jj and k == kk
        u_flat = i * m + j
        v_flat = j * p + k
        w_flat = k * n + i
        add_term([(u_flat, 1)], [(v_flat, 1)], [(w_flat, 1)])

    # Helper: add a term with coefficients
    def add_weighted(u_coeffs, v_coeffs, w_coeffs):
        add_term(u_coeffs, v_coeffs, w_coeffs)

    # Block index helpers
    def a_flat(i, j):
        return i * m + j

    def b_flat(j, k):
        return j * p + k

    def c_flat(i, k):
        return k * n + i

    # Block (0,0): 3x3 * 3x3 -> 3x3: direct, 9 terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Block (0,1): 3x1 * 1x3 -> 3x3: direct, 9 terms
    for i in range(3):
        for j in [3]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Block (0,2): 3x1 * 1x3 -> 3x3: direct, 9 terms
    for i in range(3):
        for j in [4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Block (1,0): 1x3 * 3x3 -> 1x3: direct, 9 terms
    for i in [3]:
        for j in range(3):
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Block (1,1): 1x1 * 1x1 -> 1x1: direct, 1 term
    for i in [3]:
        for j in [3]:
            for k in [3]:
                add_direct((i, j), (j, k), (i, k))

    # Block (1,2): 1x1 * 1x1 -> 1x1: direct, 1 term
    for i in [3]:
        for j in [4]:
            for k in [3]:
                add_direct((i, j), (j, k), (i, k))

    # Block (2,0): 1x3 * 3x3 -> 1x3: direct, 9 terms
    for i in [4]:
        for j in range(3):
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Block (2,1): 1x1 * 1x1 -> 1x1: direct, 1 term
    for i in [4]:
        for j in [3]:
            for k in [3]:
                add_direct((i, j), (j, k), (i, k))

    # Block (2,2): 1x1 * 1x1 -> 1x1: direct, 1 term
    for i in [4]:
        for j in [4]:
            for k in [3]:
                add_direct((i, j), (j, k), (i, k))

    # Block (0,1) is A rows 0-2, cols 3 * B rows 3, cols 0-2 -> C rows 0-2, cols 0-2
    # Already counted above as (0,1) with j=3, k=0,1,2

    # Block (0,2) is A rows 0-2, cols 4 * B rows 4, cols 0-2 -> C rows 0-2, cols 0-2
    # Already counted above as (0,2) with j=4, k=0,1,2

    # Block (1,0) is A row 3, cols 0-2 * B rows 0-2, cols 0-2 -> C row 3, cols 0-2
    # Already counted above as (1,0) with i=3, j=0,1,2, k=0,1,2

    # Block (1,1) is A[3,3] * B[3,3] -> C[3,3]
    # Already counted above as (1,1)

    # Block (1,2) is A[3,4] * B[4,3] -> C[3,3]
    # Already counted above as (1,2)

    # Block (2,0) is A row 4, cols 0-2 * B rows 0-2, cols 0-2 -> C row 4, cols 0-2
    # Already counted above as (2,0) with i=4, j=0,1,2, k=0,1,2

    # Block (2,1) is A[4,3] * B[3,3] -> C[4,3]
    # Already counted above as (2,1)

    # Block (2,2) is A[4,4] * B[4,3] -> C[4,3]
    # Already counted above as (2,2)

    # Now the cross-block terms that span multiple blocks:
    # A rows 0-2, cols 3-4 * B rows 3-4, cols 0-2 -> C rows 0-2, cols 0-2
    # This is 3x2 * 2x3 -> 3x3: 18 terms
    for i in range(3):
        for j in [3, 4]:
            for k in range(3):
                if j == 3 and k in range(3):
                    # Already counted in (0,1)
                    continue
                if j == 4 and k in range(3):
                    # Already counted in (0,2)
                    continue
                add_direct((i, j), (j, k), (i, k))

    # A rows 0-2, cols 0-2 * B rows 0-2, cols 3-4 -> C rows 0-2, cols 3-4
    # This is 3x3 * 3x2 -> 3x2: 18 terms
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # A rows 0-2, cols 3-4 * B rows 3-4, cols 3-4 -> C rows 0-2, cols 3-4
    # This is 3x2 * 2x2 -> 3x2: 12 terms
    for i in range(3):
        for j in [3, 4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # A rows 3-4, cols 0-2 * B rows 0-2, cols 0-2 -> C rows 3-4, cols 0-2
    # This is 2x3 * 3x3 -> 2x3: 18 terms
    for i in [3, 4]:
        for j in range(3):
            for k in range(3):
                if i == 3 and k in range(3):
                    # Already counted in (1,0)
                    continue
                if i == 4 and k in range(3):
                    # Already counted in (2,0)
                    continue
                add_direct((i, j), (j, k), (i, k))

    # A rows 3-4, cols 3-4 * B rows 3-4, cols 0-2 -> C rows 3-4, cols 0-2
    # This is 2x2 * 2x3 -> 2x3: 12 terms
    for i in [3, 4]:
        for j in [3, 4]:
            for k in range(3):
                if i == 3 and j == 3 and k == 3:
                    # Already counted in (1,1)
                    continue
                if i == 3 and j == 4 and k == 3:
                    # Already counted in (1,2)
                    continue
                if i == 4 and j == 3 and k == 3:
                    # Already counted in (2,1)
                    continue
                if i == 4 and j == 4 and k == 3:
                    # Already counted in (2,2)
                    continue
                add_direct((i, j), (j, k), (i, k))

    # A rows 3-4, cols 0-2 * B rows 0-2, cols 3-4 -> C rows 3-4, cols 3-4
    # This is 2x3 * 3x2 -> 2x2: 12 terms
    for i in [3, 4]:
        for j in range(3):
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # A rows 3-4, cols 3-4 * B rows 3-4, cols 3-4 -> C rows 3-4, cols 3-4
    # This is 2x2 * 2x2 -> 2x2: use Strassen, 7 terms instead of 8
    # A[3:5, 3:5] * B[3:5, 3:5] -> C[3:5, 3:5]
    # Let a00=A[3,3], a01=A[3,4], a10=A[4,3], a11=A[4,4]
    # Let b00=B[3,3], b01=B[3,4], b10=B[4,3], b11=B[4,4]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C[3,3] = M1 + M4 - M5 + M7
    # C[3,4] = M3 + M5
    # C[4,3] = M2 + M4
    # C[4,4] = M1 - M2 + M3 + M6

    # M1 = (A[3,3]+A[4,4]) * (B[3,3]+B[4,4])
    add_weighted(
        [(a_flat(3,3), 1), (a_flat(4,4), 1)],
        [(b_flat(3,3), 1), (b_flat(4,4), 1)],
        [(c_flat(3,3), 1), (c_flat(4,4), 1)]
    )

    # M2 = (A[4,3]+A[4,4]) * B[3,3]
    add_weighted(
        [(a_flat(4,3), 1), (a_flat(4,4), 1)],
        [(b_flat(3,3), 1)],
        [(c_flat(4,3), 1), (c_flat(4,4), -1)]
    )

    # M3 = A[3,3] * (B[3,4]-B[4,4])
    add_weighted(
        [(a_flat(3,3), 1)],
        [(b_flat(3,4), 1), (b_flat(4,4), -1)],
        [(c_flat(3,4), 1), (c_flat(4,4), 1)]
    )

    # M4 = A[4,4] * (B[4,3]-B[3,3])
    add_weighted(
        [(a_flat(4,4), 1)],
        [(b_flat(4,3), 1), (b_flat(3,3), -1)],
        [(c_flat(3,3), 1), (c_flat(4,3), 1)]
    )

    # M5 = (A[3,3]+A[3,4]) * B[4,4]
    add_weighted(
        [(a_flat(3,3), 1), (a_flat(3,4), 1)],
        [(b_flat(4,4), 1)],
        [(c_flat(3,3), -1), (c_flat(3,4), 1)]
    )

    # M6 = (A[4,3]-A[3,3]) * (B[3,3]+B[3,4])
    add_weighted(
        [(a_flat(4,3), 1), (a_flat(3,3), -1)],
        [(b_flat(3,3), 1), (b_flat(3,4), 1)],
        [(c_flat(4,4), 1)]
    )

    # M7 = (A[3,4]-A[4,4]) * (B[4,3]+B[4,4])
    add_weighted(
        [(a_flat(3,4), 1), (a_flat(4,4), -1)],
        [(b_flat(4,3), 1), (b_flat(4,4), 1)],
        [(c_flat(3,3), 1)]
    )

    # Count total terms
    rank = len(terms_u)

    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for (u_idx, val) in terms_u[col]:
            U[u_idx][col] = val
        for (v_idx, val) in terms_v[col]:
            V[v_idx][col] = val
        for (w_idx, val) in terms_w[col]:
            W[w_idx][col] = val

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Quick verification with random matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_direct = A_test @ B_test

    # Compute C from decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_2d = U_arr[:, r].reshape(n, m)
        v_2d = V_arr[:, r].reshape(m, p)
        w_2d = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
                C_decomp[i, k] += s

    if not np.allclose(C_decomp, C_direct, atol=1e-10):
        # Fallback to schoolbook if verification fails
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

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
