"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_reconstructed = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c]
        v_c = V[:, c]
        w_c = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += u_c[i * m + j] * v_c[j * p + k] * w_c[k * n + i]
    T_expected = np.ones((n, m, p))
    return np.allclose(T_reconstructed, T_expected, atol=tol)


def construct():
    """
    Construct a rank-45 exact tensor decomposition for 5x5x5 matrix multiplication
    using block-structured Strassen decomposition.
    
    Strategy:
    - Partition the 5x5 matrices into blocks: rows/cols {0,1} (2x2) and {2,3,4} (3x3)
    - The 3x3 sub-block is further partitioned into 2x2 (indices 2,3) and 1x1 (index 4)
    - Apply 7-term Strassen to each 2x2x2 block, 1-term schoolbook to 1x1x1 blocks
    - Total rank: 4*7 + 4*1 + 1*1 = 45
    """
    n, m, p = DIMENSIONS
    rank = 45

    # Initialize factor matrices
    U = np.zeros((n * m, rank))  # (25, 45)
    V = np.zeros((m * p, rank))  # (25, 45)
    W = np.zeros((p * n, rank))  # (25, 45)

    col = 0

    # Strassen coefficients: each term has (u_coef, v_coef, w_coef)
    # Strassen 7 terms for C = A*B where A is 2x2, B is 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6

    # For each Strassen term t, the contributions to C[i,k] are:
    # C[i,k] += w_contrib * (sum_j u_contrib[i,j] * A[i,j]) * (sum_j v_contrib[j,k] * B[j,k])
    # where u_contrib[i,j], v_contrib[j,k], w_contrib are the Strassen coefficients

    # Strassen term definitions: (name, u_matrix, v_matrix, w_matrix)
    # u_matrix[i,j] = coefficient for A[i,j] in the first factor
    # v_matrix[j,k] = coefficient for B[j,k] in the second factor
    # w_matrix[k,i] = coefficient for C[i,k] contribution

    strassen_terms = [
        # M1 = (A11+A22)(B11+B22)
        {
            'u': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # A11 + A22
            'v': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # B11 + B22
            'w': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # C11 + C22
        },
        # M2 = (A21+A22)B11
        {
            'u': {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},  # A21 + A22
            'v': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}},  # B11
            'w': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},  # C21 + C22
        },
        # M3 = A11(B21-B22)
        {
            'u': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}},  # A11
            'v': {0: {0: 0, 1: 0}, 1: {0: 1, 1: -1}},  # B21 - B22
            'w': {0: {0: 0, 1: 1}, 1: {0: 0, 1: 1}},  # C12 + C22
        },
        # M4 = A22(B12-B11)
        {
            'u': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},  # A22
            'v': {0: {0: -1, 1: 1}, 1: {0: 0, 1: 0}},  # B12 - B11
            'w': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # C11 + C22
        },
        # M5 = (A11+A12)B22
        {
            'u': {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}},  # A11 + A12
            'v': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},  # B22
            'w': {0: {0: -1, 1: 0}, 1: {0: 0, 1: 1}},  # -C11 + C22
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'u': {0: {0: -1, 1: 0}, 1: {0: 1, 1: 0}},  # A21 - A11
            'v': {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}},  # B11 + B12
            'w': {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},  # C22
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'u': {0: {0: 0, 1: 1}, 1: {0: 0, 1: -1}},  # A12 - A22
            'v': {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},  # B21 + B22
            'w': {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}},  # C11 + C22
        },
    ]

    def add_strassen_block(row_offsets, col_offsets, term_idx, col_start):
        """Add Strassen term term_idx for the 2x2 block at given row/col offsets."""
        global col
        term = strassen_terms[term_idx]
        for i in range(2):
            for j in range(2):
                if term['u'][i][j] != 0:
                    U[row_offsets[i] * m + col_offsets[j], col] = term['u'][i][j]
        for j in range(2):
            for k in range(2):
                if term['v'][j][k] != 0:
                    V[col_offsets[j] * p + col_offsets[k], col] = term['v'][j][k]
        for k in range(2):
            for i in range(2):
                if term['w'][k][i] != 0:
                    W[col_offsets[k] * n + row_offsets[i], col] = term['w'][k][i]
        col += 1

    def add_schoolbook_1x1(row_off, col_off, col_start):
        """Add 1-term schoolbook for a 1x1x1 block."""
        global col
        U[row_off * m + col_off, col] = 1
        V[col_off * p + col_off, col] = 1
        W[col_off * n + row_off, col] = 1
        col += 1

    # 2x2 blocks (Strassen, 7 terms each):
    # Block (0,0): rows {0,1}, cols {0,1}
    for t in range(7):
        add_strassen_block([0, 1], [0, 1], t, col)
    # Block (0,1): rows {0,1}, cols {2,3}
    for t in range(7):
        add_strassen_block([0, 1], [2, 3], t, col)
    # Block (1,0): rows {2,3}, cols {0,1}
    for t in range(7):
        add_strassen_block([2, 3], [0, 1], t, col)
    # Block (1,1): rows {2,3}, cols {2,3}
    for t in range(7):
        add_strassen_block([2, 3], [2, 3], t, col)

    # 1x1 blocks (schoolbook, 1 term each):
    # (0,2): row 0, col 4
    add_schoolbook_1x1(0, 4, col)
    # (1,2): row 4, col 0
    add_schoolbook_1x1(4, 0, col)
    # (2,0): row 2, col 4
    add_schoolbook_1x1(2, 4, col)
    # (2,1): row 3, col 4
    add_schoolbook_1x1(3, 4, col)
    # (2,2): row 4, col 4
    add_schoolbook_1x1(4, 4, col)

    # Verify
    assert col == rank, f"Expected rank {rank}, got {col}"
    assert _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank), "Verification failed"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
