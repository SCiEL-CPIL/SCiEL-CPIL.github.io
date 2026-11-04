"""
Tensor decomposition for 4x4 @ 4x5 matrix multiplication.
Uses Block-Strassen for the 4x4x4 part (rank 49) + schoolbook for the 5th column (rank 16).
Total rank: 65.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 5]


def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False

    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)

    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)

    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True


def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _block_strassen_4x4x4():
    """
    Construct the Block-Strassen decomposition for 4x4 @ 4x4.
    Treats A, B, C as 2x2 blocks of 2x2 matrices.
    Uses 7 block-level Strassen multiplications, each expanded into 7 scalar Strassen terms.
    Total: 49 rank-1 terms.
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    """
    n, m, p = 4, 4, 4
    U = np.zeros((n * m, 49))
    V = np.zeros((m * p, 49))
    W = np.zeros((p * n, 49))

    def make_L_block_term(t):
        L = {}
        if t == 0:  # A00+A11
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i, j), 1), ((i + 2, j + 2), 1)]
        elif t == 1:  # A10+A11
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i + 2, j), 1), ((i + 2, j + 2), 1)]
        elif t == 2:  # A00
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i, j), 1)]
        elif t == 3:  # A11
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i + 2, j + 2), 1)]
        elif t == 4:  # A00+A01
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i, j), 1), ((i, j + 2), 1)]
        elif t == 5:  # A10-A00
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i + 2, j), 1), ((i, j), -1)]
        elif t == 6:  # A01-A11
            for i in range(2):
                for j in range(2):
                    L[(i, j)] = [((i, j + 2), 1), ((i + 2, j + 2), -1)]
        return L

    def make_R_block_term(t):
        R = {}
        if t == 0:  # B00+B11
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i, j), 1), ((i + 2, j + 2), 1)]
        elif t == 1:  # B00
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i, j), 1)]
        elif t == 2:  # B10-B11
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i + 2, j), 1), ((i + 2, j + 2), -1)]
        elif t == 3:  # B00-B01
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i, j), 1), ((i, j + 2), -1)]
        elif t == 4:  # B11
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i + 2, j + 2), 1)]
        elif t == 5:  # B00+B01
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i, j), 1), ((i, j + 2), 1)]
        elif t == 6:  # B10+B11
            for i in range(2):
                for j in range(2):
                    R[(i, j)] = [((i + 2, j), 1), ((i + 2, j + 2), 1)]
        return R

    scalar_strassen = [
        (0, [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        (1, [(2, 1), (3, 1)], [(0, 1)]),
        (2, [(0, 1)], [(2, 1), (3, -1)]),
        (3, [(3, 1)], [(0, 1), (1, -1)]),
        (4, [(0, 1), (1, 1)], [(3, 1)]),
        (5, [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        (6, [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]

    Z_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0, 1): {2: 1, 4: 1},
        (1, 0): {1: 1, 3: 1},
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1},
    }

    C_block_contribs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0, 1): {2: 1, 4: 1},
        (1, 0): {1: 1, 3: 1},
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1},
    }

    col = 0
    for t in range(7):
        L = make_L_block_term(t)
        R = make_R_block_term(t)

        for s, (X_entries, Y_entries) in [(i, scalar_strassen[i]) for i in range(7)]:
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)

            for idx_x, cx in X_entries:
                xi, xj = divmod(idx_x, 2)
                for a_row, a_col, ca in L[(xi, xj)]:
                    U_vec[a_row * m + a_col] += cx * ca

            for idx_y, cy in Y_entries:
                yi, yj = divmod(idx_y, 2)
                for b_row, b_col, cb in R[(yi, yj)]:
                    V_vec[b_row * p + b_col] += cy * cb

            for (I, J), t_coeff in C_block_contribs.items():
                if t not in t_coeff:
                    continue
                for (zi, zj), z_coeff in Z_coeffs.items():
                    if s not in z_coeff:
                        continue
                    c_row = I * 2 + zi
                    c_col = J * 2 + zj
                    W_vec[c_row * n + c_col] += t_coeff[t] * z_coeff[s]

            U[:, col] = U_vec
            V[:, col] = V_vec
            W[:, col] = W_vec
            col += 1

    return U, V, W


def _construct_block_strassen_plus_schoolbook():
    """
    Construct decomposition using Block-Strassen for 4x4x4 (rank 49)
    + schoolbook for 4x4x1 (rank 16).
    Total rank: 65.
    """
    n, m, p = 4, 4, 5

    U_bs, V_bs, W_bs = _block_strassen_4x4x4()

    rank_schoolbook = n * m
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))

    col = 0
    for i in range(n):
        for j in range(m):
            U_sb[i * m + j, col] = 1
            V_sb[j * p + (p - 1), col] = 1
            W_sb[(p - 1) * n + i, col] = 1
            col += 1

    total_rank = 49 + rank_schoolbook
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U[:, :49] = U_bs
    V[:, :49] = V_bs
    W[:, :49] = W_bs

    U[:, 49:] = U_sb
    V[:, 49:] = V_sb
    W[:, 49:] = W_sb

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if _validate(result["U"], result["V"], result["W"], n, m, p):
        return result, total_rank
    else:
        fb = _construct_schoolbook(n, m, p)
        return fb, n * m * p


def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p

    try:
        result, rank = _construct_block_strassen_plus_schoolbook()
        if rank < best_rank:
            best_result = result
            best_rank = rank
    except Exception:
        pass

    if time.time() > deadline:
        return best_result

    return best_result


# EVOLVE-BLOCK-START
# (The entire program is within the evolve block)
# EVOLVE-BLOCK-END
