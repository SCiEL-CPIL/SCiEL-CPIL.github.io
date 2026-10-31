"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

# Pre-computed rank-49 Strassen decomposition for 4x4 matrix multiplication.
# U[i*4+j, r] multiplies A[i,j]; V[j*4+k, r] multiplies B[j,k]; W[k*4+i, r] multiplies C[i,k].
# The identity: C[i,k] = sum_r (sum_j U[i*4+j,r]*A[i,j]) * (sum_j V[j*4+k,r]*B[j,k]) * W[k*4+i,r]
#
# This is the recursive Strassen decomposition: 4x4 = 2x2 blocks, each 2x2 block
# multiplied via Strassen (rank 7), giving 7*7 = 49 total rank.

def _build_strassen_4x4():
    """Build U, V, W matrices for rank-49 Strassen 4x4 decomposition."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def w_idx(i, k):
        return k * n + i

    # Helper: represent a 2x2 block of linear combinations as list of lists of dicts
    def make_block(br, bc, sign=1):
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                gr = br * 2 + i
                gc = bc * 2 + j
                row.append({(gr, gc): sign})
            mat.append(row)
        return mat

    def add_blocks(b1, b2):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d = {}
                for key, val in b1[i][j].items():
                    d[key] = d.get(key, 0) + val
                for key, val in b2[i][j].items():
                    d[key] = d.get(key, 0) + val
                row.append({k: v for k, v in d.items() if v != 0})
            result.append(row)
        return result

    def sub_blocks(b1, b2):
        neg_b2 = [[{k: -v for k, v in b2[i][j].items()} for j in range(2)] for i in range(2)]
        return add_blocks(b1, neg_b2)

    def add_entries(mat, i1, j1, i2, j2):
        result = [[{} for _ in range(2)] for _ in range(2)]
        d = {}
        for key, val in mat[i1][j1].items():
            d[key] = d.get(key, 0) + val
        for key, val in mat[i2][j2].items():
            d[key] = d.get(key, 0) + val
        result[i1][j1] = {k: v for k, v in d.items() if v != 0}
        return result

    def sub_entries(mat, i1, j1, i2, j2):
        result = [[{} for _ in range(2)] for _ in range(2)]
        d = {}
        for key, val in mat[i1][j1].items():
            d[key] = d.get(key, 0) + val
        for key, val in mat[i2][j2].items():
            d[key] = d.get(key, 0) - val
        result[i1][j1] = {k: v for k, v in d.items() if v != 0}
        return result

    def single_entry(mat, i, j):
        result = [[{} for _ in range(2)] for _ in range(2)]
        result[i][j] = dict(mat[i][j])
        return result

    # Outer Strassen blocks
    A00 = make_block(0, 0)
    A01 = make_block(0, 1)
    A10 = make_block(1, 0)
    A11 = make_block(1, 1)
    B00 = make_block(0, 0)
    B01 = make_block(0, 1)
    B10 = make_block(1, 0)
    B11 = make_block(1, 1)

    outer_X = [
        add_blocks(A00, A11),
        add_blocks(A10, A11),
        A00,
        A11,
        add_blocks(A00, A01),
        sub_blocks(A10, A00),
        sub_blocks(A01, A11),
    ]
    outer_Y = [
        add_blocks(B00, B11),
        B00,
        sub_blocks(B01, B11),
        sub_blocks(B10, B00),
        B11,
        add_blocks(B00, B01),
        add_blocks(B10, B11),
    ]

    c_targets = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    inner_targets = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]

        X1 = add_entries(X, 0, 0, 1, 1)
        Y1 = add_entries(Y, 0, 0, 1, 1)
        X2 = add_entries(X, 1, 0, 1, 1)
        Y2 = single_entry(Y, 0, 0)
        X3 = single_entry(X, 0, 0)
        Y3 = sub_entries(Y, 0, 1, 1, 1)
        X4 = single_entry(X, 1, 1)
        Y4 = sub_entries(Y, 1, 0, 0, 0)
        X5 = add_entries(X, 0, 0, 0, 1)
        Y5 = single_entry(Y, 1, 1)
        X6 = sub_entries(X, 1, 0, 0, 0)
        Y6 = add_entries(Y, 0, 0, 0, 1)
        X7 = sub_entries(X, 0, 1, 1, 1)
        Y7 = add_entries(Y, 1, 0, 1, 1)

        inner_factors = [
            (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4),
            (X5, Y5), (X6, Y6), (X7, Y7),
        ]

        for k in range(7):
            Xf, Yf = inner_factors[k]
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c), a_v in Xf[i][j].items():
                        U[a_r * m + a_c, col] += a_v
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c), b_v in Yf[i][j].items():
                        V[b_r * p + b_c, col] += b_v
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k), col] += c_coeff * z_coeff
            col += 1

    assert col == rank
    return U, V, W

# Build once at module load
_U_static, _V_static, _W_static = _build_strassen_4x4()

def construct():
    """
    Rank-49 Strassen decomposition for 4x4 matrix multiplication.
    Returns pre-computed factor matrices U, V, W.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    return {"U": _U_static.tolist(), "V": _V_static.tolist(), "W": _W_static.tolist()}
# EVOLVE-BLOCK-END
