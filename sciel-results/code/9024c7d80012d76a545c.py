import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0

    # Strassen 2x2x2 components.
    # local A index: 0=A11,1=A12,2=A21,3=A22
    # local B index: 0=B11,1=B12,2=B21,3=B22
    # local W index: 0=C11,1=C12,2=C21,3=C22
    # Each entry: list of (local_idx, coeff)
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]

    # The 2x4 x 4x6 product decomposes into 3 column-groups of 2 columns each.
    # For column-group g (cols 2g, 2g+1 of B):
    #   C[:, 2g:2g+2] = A[:,0:2] @ B[0:2, 2g:2g+2]  +  A[:,2:4] @ B[2:4, 2g:2g+2]
    # Each term is a 2x2 x 2x2 multiplication.
    # We apply Strassen to each 2x2x2 term.
    #
    # Total rank = 3 (col groups) * 2 (A halves) * 7 (Strassen) = 42.
    #
    # Index conventions (flattened):
    #   U index = i*m + k  (A[i,k], i in 0..n-1, k in 0..m-1)  -> n*m = 8
    #   V index = k*p + j  (B[k,j], k in 0..m-1, j in 0..p-1)  -> m*p = 24
    #   W index = j*n + i  (C[i,j], i in 0..n-1, j in 0..p-1)  -> p*n = 12

    n, m, p = 2, 4, 6
    U = [[0] * 42 for _ in range(n * m)]   # 8 x 42
    V = [[0] * 42 for _ in range(m * p)]   # 24 x 42
    W = [[0] * 42 for _ in range(p * n)]   # 12 x 42

    col = 0
    for g in range(3):  # 3 column-groups
        if not time.time() < deadline:
            break
        b_cols = [2 * g, 2 * g + 1]  # the 2 output columns of this group

        # Two 2x2x2 products: A-half 0 (cols 0,1) with B-rows 0,1
        # and A-half 1 (cols 2,3) with B-rows 2,3.
        for half in range(2):
            if not time.time() < deadline:
                break
            a_cols = [2 * half, 2 * half + 1]   # A columns
            b_rows = [2 * half, 2 * half + 1]   # B rows

            # local -> global maps
            a_map = {
                0: 0 * m + a_cols[0],   # A[0, a_cols[0]]
                1: 0 * m + a_cols[1],   # A[0, a_cols[1]]
                2: 1 * m + a_cols[0],   # A[1, a_cols[0]]
                3: 1 * m + a_cols[1],   # A[1, a_cols[1]]
            }
            b_map = {
                0: b_rows[0] * p + b_cols[0],
                1: b_rows[0] * p + b_cols[1],
                2: b_rows[1] * p + b_cols[0],
                3: b_rows[1] * p + b_cols[1],
            }
            w_map = {
                0: b_cols[0] * n + 0,   # C[0, b_cols[0]]
                1: b_cols[1] * n + 0,   # C[0, b_cols[1]]
                2: b_cols[0] * n + 1,   # C[1, b_cols[0]]
                3: b_cols[1] * n + 1,   # C[1, b_cols[1]]
            }

            for u_coeffs, v_coeffs, w_coeffs in strassen:
                if not time.time() < deadline:
                    break
                for li, c in u_coeffs:
                    U[a_map[li]][col] += c
                for li, c in v_coeffs:
                    V[b_map[li]][col] += c
                for li, c in w_coeffs:
                    W[w_map[li]][col] += c
                col += 1

    return {"U": U, "V": V, "W": W}

# Self-verification (runs once at import, cheap)
def _verify():
    try:
        res = construct()
        U = np.array(res["U"], dtype=float)
        V = np.array(res["V"], dtype=float)
        W = np.array(res["W"], dtype=float)
        n, m, p = 2, 4, 6
        rank = U.shape[1]
        # Build tensor T[i,k,j] = delta(i,k) via U,V,W:
        # T = sum_r U[:,r] * V[:,r] * W[:,r]  where U indexed (i*m+k), V (k*p+j), W (j*n+i)
        # We want T[i,k,j] = 1 if (i,k,j) is a valid multiplication entry, i.e. T[i,k,j] = 1 for all i,k,j.
        # Actually the multiplication tensor has T[i,k,j] = 1 for all i in [n], k in [m], j in [p].
        # Check: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 for all i,k,j.
        T = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i, k, j] = float(np.sum(U[i * m + k, :] * V[k * p + j, :] * W[j * n + i, :]))
        expected = np.ones((n, m, p))
        ok = np.allclose(T, expected, atol=1e-9)
        return ok, rank
    except Exception as e:
        return False, -1

_ok, _rank = _verify()
# EVOLVE-BLOCK-END
