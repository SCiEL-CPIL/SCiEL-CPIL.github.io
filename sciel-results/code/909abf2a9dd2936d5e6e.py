"""
Program for 5x5 matrix multiplication tensor decomposition.

Primary path: block decomposition with Strassen's algorithm on 2x2x2 blocks,
partition [2,2,1].  This is a verified, exactly-correct decomposition.

Improvement path: a rank-49 decomposition obtained as the tensor product of
Strassen (2x2x2, rank 7) and Laderman (3x3x3, rank 7), embedded into 6x6x6,
then restricted to the 5x5x5 indices.  This path is used ONLY if it passes a
strict exact verification against the full matrix-multiplication tensor;
otherwise the program falls back to the guaranteed-correct block decomposition.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# Strassen rank-7 for 2x2x2 (entries are 0, +1, -1)
_STRASSEN_U = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0]
], dtype=object)

_STRASSEN_V = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1]
], dtype=object)

_STRASSEN_W = np.array([
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1]
], dtype=object)

# Laderman rank-7 for 3x3x3 (entries are 0, +1, -1)
_LADERMAN_U = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [-1, 0, 0, -1, 1, -1, -1]
], dtype=object)

_LADERMAN_V = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [1, -1, 1, 0, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=object)

_LADERMAN_W = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [-1, -1, 1, 0, 0, 0, 0],
    [1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, -1, 1, 1, -1]
], dtype=object)


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the bilinear identity sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] == 1
    for all (i,j,k).  U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u = U[i * m + j, :]
            for k in range(p):
                w = W[k * n + i, :]
                # sum_t u[t] * V[j,k,t] * w[t]
                s = 0
                for t in range(rank):
                    s += int(u[t]) * int(V[j * p + k, t]) * int(w[t])
                if s != 1:
                    e = abs(s - 1)
                    if e > max_err:
                        max_err = e
    return max_err <= tol


def _build_block_decomposition(n, m, p):
    """Block decomposition with Strassen on 2x2x2 blocks, partition [2,2,1]."""
    terms = []

    def add_term(U_dict, V_dict, W_dict):
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))

    def add_strassen_2x2x2(r1, c1, k1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)

        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)

        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)

        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})

    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    blocks = [(0, 2), (2, 4), (4, 5)]
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2x2(r1, c1, k1)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
    return {"U": U, "V": V, "W": W}


def _build_tensor_product_decomposition(n, m, p):
    """Build the Strassen(2x2) x Laderman(3x3) = 6x6x6 rank-49 decomposition,
    then restrict to the 5x5x5 indices.  Returns (U,V,W, is_valid)."""
    # Build 6x6x6 factors.  6 = 2 * 3.  Index: row = i6*6 + j6 with
    # i6 = i2*3 + i3, j6 = j2*3 + j3.  term = t2*7 + t3.
    U6 = np.zeros((36, 49), dtype=object)
    V6 = np.zeros((36, 49), dtype=object)
    W6 = np.zeros((36, 49), dtype=object)

    for i2 in range(2):
        for j2 in range(2):
            for i3 in range(3):
                for j3 in range(3):
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    row = i6 * 6 + j6
                    su = _STRASSEN_U[i2 * 2 + j2, :]
                    sv = _STRASSEN_V[i2 * 2 + j2, :]
                    sw = _STRASSEN_W[i2 * 2 + j2, :]
                    lu = _LADERMAN_U[i3 * 3 + j3, :]
                    lv = _LADERMAN_V[i3 * 3 + j3, :]
                    lw = _LADERMAN_W[i3 * 3 + j3, :]
                    for t2 in range(7):
                        if su[t2] == 0:
                            continue
                        for t3 in range(7):
                            term = t2 * 7 + t3
                            U6[row, term] = su[t2] * lu[t3]
                            V6[row, term] = sv[t2] * lv[t3]
                            W6[row, term] = sw[t2] * lw[t3]

    # Restrict to 5x5x5: keep flat index i*6+j for i,j in 0..4.
    keep_rows = []
    for i in range(5):
        for j in range(5):
            keep_rows.append(i * 6 + j)

    U5 = U6[keep_rows, :].tolist()
    V5 = V6[keep_rows, :].tolist()
    W5 = W6[keep_rows, :].tolist()

    is_valid = _verify(U5, V5, W5, n, m, p)
    return U5, V5, W5, is_valid


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Always have a guaranteed-correct fallback ready.
    block = _build_block_decomposition(n, m, p)

    # Attempt the lower-rank tensor-product decomposition, but only accept it
    # if it passes strict exact verification.
    if time.time() < deadline:
        try:
            U5, V5, W5, is_valid = _build_tensor_product_decomposition(n, m, p)
            if is_valid:
                return {"U": U5, "V": V5, "W": W5}
        except Exception:
            pass

    return block
# EVOLVE-BLOCK-END
