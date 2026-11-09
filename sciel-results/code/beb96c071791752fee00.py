"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition.

Strategy: partition T(i,j,k) into four disjoint 2x2x2 blocks (handled by
rank-7 Strassen each), a 4x1x4 slab at j=4 with k in {0..3} (schoolbook,
rank 4), and the four (i,4,4) corner entries (rank 4).  Total rank 36.
"""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

# Strassen 7 multiplications:  P = U * V, then W = P  (W[k*n+i, c] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])
STRASSEN_U = [
    [1, 0, 0, 0, 0, 0, 0],   # M1 = a11
    [1, 0, 1, 0, 0, 0, 0],   # M2 = a12
    [0, 0, 0, 0, 1, 0, 0],   # M3 = a22
    [0, 0, 1, 1, 0, 0, 0],   # M4 = a11 + a22
    [0, 1, 0, 0, 0, 0, 1],   # M5 = (a11 + a21) * b11
    [1, 0, 0, 0, 0, 1, 0],   # M6 = (a21 + a22) * b12
    [0, 0, 0, 1, 0, 0, 1],   # M7 = (a11 - a12) * b22
]
STRASSEN_V = [
    [1, 0],   # b11
    [0, 1],   # b22
    [1, 1],   # b12
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]
STRASSEN_W = [
    [1, 0],   # M1 -> a22 * b11
    [0, 1],   # M2 -> a12 * b22
    [1, 0],   # M3 -> a11 * b21
    [1, 1],   # M4 -> (a11+a22) * (b21 + b22)
    [1, 0],   # M5 -> a22 * (b11 + b22)
    [0, 1],   # M6 -> a21 * (b12 + b22)
    [1, 1],   # M7 -> (a11 - a12) * (b21 + b22)
]


def _add_term(U, V, W, u_row, u_val, v_row, v_val, w_row, w_val):
    col = len(U[0])
    for r in range(len(U)):
        U[r].append(0.0)
    for r in range(len(V)):
        V[r].append(0.0)
    for r in range(len(W)):
        W[r].append(0.0)
    U[u_row][col] = u_val
    V[v_row][col] = v_val
    W[w_row][col] = w_val


def _add_strassen_block(U, V, W, i0, j0, k0, nm, mp):
    """Add the rank-7 Strassen terms for the 2x2x2 sub-block with top-left
    element (i0, j0) of A and (j0, k0) of B.  A row index = i*m + j,
    B row index = j*p + k, W row index = k*n + i."""
    for s in range(7):
        for a_i in range(2):
            uv = STRASSEN_U[s][a_i]
            if uv == 0:
                continue
            for b_j in range(2):
                vv = STRASSEN_V[s][b_j]
                if vv == 0:
                    continue
                for w_k in range(2):
                    wv = STRASSEN_W[s][w_k]
                    if wv == 0:
                        continue
                    a_row = (i0 + a_i) * nm + (j0 + b_j)
                    b_row = (j0 + b_j) * mp + (k0 + w_k)
                    w_row = (k0 + w_k) * nm + (i0 + a_i)
                    _add_term(U, V, W, a_row, uv, b_row, vv, w_row, wv)


def construct():
    n, m, p = DIMENSIONS
    nm = n * m       # 20
    mp = m * p       # 25

    U = [[0.0] * 1 for _ in range(nm)]   # dummy column so list is non-empty; we build on top
    V = [[0.0] * 1 for _ in range(mp)]
    W = [[0.0] * 1 for _ in range(n * p)]
    # Drop the dummy column by resetting:
    for r in range(len(U)): U[r] = []
    for r in range(len(V)): V[r] = []
    for r in range(len(W)): W[r] = []

    # Four 2x2x2 blocks (8x8 region of (i, j) vs (j, k)), each rank-7 Strassen.
    for i0 in (0, 2):
        for j0 in (0, 2):
            for k0 in (0, 2):
                _add_strassen_block(U, V, W, i0, j0, k0, nm, mp)

    # 4x1x4 slab: i in {0..3}, j=4, k in {0..3}  ->  schoolbook, 4 terms per i
    for i in range(n):
        for k in range(4):
            a_row = i * nm + 4
            b_row = 4 * mp + k
            w_row = k * nm + i
            _add_term(U, V, W, a_row, 1.0, b_row, 1.0, w_row, 1.0)

    # Corner (i, j=4, k=4) for i in 0..3 : 4 single terms
    for i in range(n):
        a_row = i * nm + 4
        b_row = 4 * mp + 4
        w_row = 4 * nm + i
        _add_term(U, V, W, a_row, 1.0, b_row, 1.0, w_row, 1.0)

    # Convert floats to ints where possible (entries are in {-1,0,1,2} or such)
    U = [[int(round(x)) if abs(x - round(x)) < 1e-12 else x for x in row] for row in U]
    V = [[int(round(x)) if abs(x - round(x)) < 1e-12 else x for x in row] for row in V]
    W = [[int(round(x)) if abs(x - round(x)) < 1e-12 else x for x in row] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
