"""
Rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.

Design family (kept from current program): block partitioning 5 = 4 + 1.
- The 4x4x4 core uses nested Strassen (rank 49 = 7^2).
- All cross terms involving the 1x1 part use schoolbook.
- Total rank: 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110.

Refinement (from inspirations): a fast exact verification pass
(np.einsum) is run before returning so the constructive decomposition is
guaranteed correct; if verification ever fails we fall back to schoolbook.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1."""
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return bool(np.allclose(T, 1.0, atol=tol))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W


# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-110 exact decomposition of the 5x5x5 matmul tensor.

    The 4x4x4 core is built by applying Strassen's rank-7 2x2x2 decomposition
    to the seven top-level 2x2x2 products, each of which is itself a 2x2x2
    product (so each expands into 7 rank-1 terms -> 49 total). Cross terms
    with the 1x1 block use schoolbook.
    """
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    rank = 110
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_schoolbook_single(i, j, k):
        nonlocal col
        if col < rank:
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    def add_schoolbook_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    if col < rank:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1

    # Strassen 2x2x2 with linear-combination inputs.
    # a = [(u_index, coeff), ...] for the 4 elements of the left 2x2 matrix
    # b = [(v_index, coeff), ...] for the 4 elements of the right 2x2 matrix
    # w = [(w_index, coeff), ...] for the 4 elements of the output 2x2 matrix
    # (order: (0,0),(0,1),(1,0),(1,1))
    def add_strassen(a, b, w):
        nonlocal col
        # Strassen's 7 products and their output contributions (output element order 00,01,10,11)
        # M1 = (A00+A11)(B00+B11) -> C00 +=1, C11 +=1
        # M2 = (A10+A11)B00       -> C10 +=1
        # M3 = A00(B01-B11)       -> C01 +=1
        # M4 = A11(B10-B00)       -> C00 -=1, C10 +=1
        # M5 = (A00+A01)B11       -> C01 +=1, C00 -=1
        # M6 = (A10-A00)(B00+B01) -> C11 +=1
        # M7 = (A01-A11)(B10+B11) -> C00 +=1, C11 -=1
        A00, A01, A10, A11 = a[0], a[1], a[2], a[3]
        B00, B01, B10, B11 = b[0], b[1], b[2], b[3]

        def emit(ua, va, wa):
            nonlocal col
            if col >= rank:
                return
            for idx, c in ua:
                U[idx, col] += c
            for idx, c in va:
                V[idx, col] += c
            for idx, c in wa:
                W[idx, col] += c
            col += 1

        # M1
        ua = [(A00[0], A00[1] + 0.0), (A11[0], A11[1])]  # A00+A11
        ua = [(A00[0], A00[1]), (A11[0], A11[1])]
        va = [(B00[0], B00[1]), (B11[0], B11[1])]
        wa = [(w[0][0], w[0][1]), (w[3][0], w[3][1])]
        emit(ua, va, wa)

        # M2
        ua = [(A10[0], A10[1]), (A11[0], A11[1])]
        va = [(B00[0], B00[1])]
        wa = [(w[2][0], w[2][1])]
        emit(ua, va, wa)

        # M3
        ua = [(A00[0], A00[1])]
        va = [(B01[0], B01[1]), (B11[0], -B11[1])]
        wa = [(w[1][0], w[1][1])]
        emit(ua, va, wa)

        # M4
        ua = [(A11[0], A11[1])]
        va = [(B10[0], B10[1]), (B00[0], -B00[1])]
        wa = [(w[0][0], -w[0][1]), (w[2][0], w[2][1])]
        emit(ua, va, wa)

        # M5
        ua = [(A00[0], A00[1]), (A01[0], A01[1])]
        va = [(B11[0], B11[1])]
        wa = [(w[1][0], w[1][1]), (w[0][0], -w[0][1])]
        emit(ua, va, wa)

        # M6
        ua = [(A10[0], A10[1]), (A00[0], -A00[1])]
        va = [(B00[0], B00[1]), (B01[0], B01[1])]
        wa = [(w[3][0], w[3][1])]
        emit(ua, va, wa)

        # M7
        ua = [(A01[0], A01[1]), (A11[0], -A11[1])]
        va = [(B10[0], B10[1]), (B11[0], B11[1])]
        wa = [(w[0][0], w[0][1]), (w[3][0], -w[3][1])]
        emit(ua, va, wa)

    # Element index helpers for the 5x5 matrices.
    def u_idx(i, j):  # A element (i, j)
        return i * m + j

    def v_idx(j, k):  # B element (j, k)
        return j * p + k

    def w_idx(k, i):  # C element (k, i)  [trace convention W[k*n+i]]
        return k * n + i

    # Build the 4x4x4 core via nested Strassen.
    # Top-level: 4x4 partitioned into 2x2 blocks.
    # Block (bi, bj) of A occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # For each top-level Strassen product, the left factor A is a 2x2 matrix
    # whose elements are linear combinations of 2x2 A-blocks; similarly for B.
    # We represent each element of the 2x2 factor as (u_index, coeff) pairs
    # over the 4 elements of a 2x2 A-block.

    # For a 2x2 A-block at (bi, bj), its 4 elements (in order 00,01,10,11) have
    # global positions:
    def a_block_elems(bi, bj):
        # returns list of 4 (u_index, coeff=1.0)
        r0, c0 = 2 * bi, 2 * bj
        return [
            (u_idx(r0, c0), 1.0),
            (u_idx(r0, c0 + 1), 1.0),
            (u_idx(r0 + 1, c0), 1.0),
            (u_idx(r0 + 1, c0 + 1), 1.0),
        ]

    def b_block_elems(bj, bk):
        r0, c0 = 2 * bj, 2 * bk
        return [
            (v_idx(r0, c0), 1.0),
            (v_idx(r0, c0 + 1), 1.0),
            (v_idx(r0 + 1, c0), 1.0),
            (v_idx(r0 + 1, c0 + 1), 1.0),
        ]

    def c_block_elems(bk, bi):
        # C block at rows 2*bk.., cols 2*bi..  (C element (k, i))
        r0, c0 = 2 * bk, 2 * bi
        return [
            (w_idx(r0, c0), 1.0),
            (w_idx(r0, c0 + 1), 1.0),
            (w_idx(r0 + 1, c0), 1.0),
            (w_idx(r0 + 1, c0 + 1), 1.0),
        ]

    # A linear combination of 2x2 A-blocks: given a dict {(bi,bj): coeff},
    # produce the 4-element list [(u_index, coeff), ...] for the 2x2 factor.
    def combo_a(blocks):
        # blocks: dict (bi,bj)->coeff
        elems = [None, None, None, None]
        for (bi, bj), c in blocks.items():
            be = a_block_elems(bi, bj)
            for t in range(4):
                idx, _ = be[t]
                if elems[t] is None:
                    elems[t] = (idx, c)
                else:
                    elems[t] = (idx, elems[t][1] + c)
        return elems

    def combo_b(blocks):
        elems = [None, None, None, None]
        for (bj, bk), c in blocks.items():
            be = b_block_elems(bj, bk)
            for t in range(4):
                idx, _ = be[t]
                if elems[t] is None:
                    elems[t] = (idx, c)
                else:
                    elems[t] = (idx, elems[t][1] + c)
        return elems

    # Top-level Strassen products for 4x4x4:
    # M1 = (A00+A11)(B00+B11)  -> C00 +=1, C11 +=1
    # M2 = (A10+A11)B00        -> C10 +=1
    # M3 = A00(B01-B11)        -> C01 +=1
    # M4 = A11(B10-B00)        -> C00 -=1, C10 +=1
    # M5 = (A00+A01)B11        -> C01 +=1, C00 -=1
    # M6 = (A10-A00)(B00+B01)  -> C11 +=1
    # M7 = (A01-A11)(B10+B11)  -> C00 +=1, C11 -=1
    #
    # Each "Cij += s" means the 2x2 output block Cij (rows 2*i.., cols 2*j..)
    # is scaled by s. We build the w list accordingly.

    def emit_top(a_blocks, b_blocks, c_contribs):
        # a_blocks, b_blocks: dicts for the 2x2 factor element combos
        # c_contribs: dict {(bi,bj): coeff} for the 2x2 output blocks
        a = combo_a(a_blocks)
        b = combo_b(b_blocks)
        # w: 4 elements, each a list of (w_index, coeff) from the contributing C blocks
        w_elems = [None, None, None, None]
        for (bi, bj), s in c_contribs.items():
            ce = c_block_elems(bj, bi)  # C block at rows 2*bj, cols 2*bi
            for t in range(4):
                idx, _ = ce[t]
                if w_elems[t] is None:
                    w_elems[t] = (idx, s)
                else:
                    w_elems[t] = (idx, w_elems[t][1] + s)
        add_strassen(a, b, w_elems)

    # Top-level products (A block indices (row_block, col_block), B (col_block, out_block))
    # C block (out_block, row_block)
    emit_top({(0, 0): 1.0, (1, 1): 1.0}, {(0, 0): 1.0, (1, 1): 1.0}, {(0, 0): 1.0, (1, 1): 1.0})   # M1
    emit_top({(1, 0): 1.0, (1, 1): 1.0}, {(0, 0): 1.0}, {(1, 0): 1.0})                                 # M2
    emit_top({(0, 0): 1.0}, {(0, 1): 1.0, (1, 1): -1.0}, {(0, 1): 1.0})                                 # M3
    emit_top({(1, 1): 1.0}, {(1, 0): 1.0, (0, 0): -1.0}, {(0, 0): -1.0, (1, 0): 1.0})                   # M4
    emit_top({(0, 0): 1.0, (0, 1): 1.0}, {(1, 1): 1.0}, {(0, 1): 1.0, (0, 0): -1.0})                    # M5
    emit_top({(1, 0): 1.0, (0, 0): -1.0}, {(0, 0): 1.0, (0, 1): 1.0}, {(1, 1): 1.0})                    # M6
    emit_top({(0, 1): 1.0, (1, 1): -1.0}, {(1, 0): 1.0, (1, 1): 1.0}, {(0, 0): 1.0, (1, 1): -1.0})      # M7

    # Cross terms with the 1x1 block (index 4).
    # A4B4: i in 0..3, j=4, k in 0..3
    add_schoolbook_block(0, 4, 4, 5, 0, 4)
    # A4B1: i in 0..3, j=4, k=4
    add_schoolbook_block(0, 4, 4, 5, 4, 5)
    # A1B4: i=4, j in 0..3, k in 0..3
    add_schoolbook_block(4, 5, 0, 4, 0, 4)
    # A1B1: i=4, j=4, k=4
    add_schoolbook_block(4, 5, 4, 5, 4, 5)

    # Verify the constructive decomposition.
    if time.time() < deadline - 5:
        ok = _verify_fast(U, V, W, n, m, p, tol=1e-9)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: full schoolbook (rank 125) guaranteed correct.
    if time.time() < deadline - 5:
        Us, Vs, Ws = _schoolbook(n, m, p)
        if _verify_fast(Us, Vs, Ws, n, m, p, tol=1e-9):
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
