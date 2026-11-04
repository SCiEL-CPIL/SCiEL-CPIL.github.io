"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Block tiling with 2x2 A-blocks and 2x3 B-blocks.
Each output 2x3 block = (2x2)*(2x3) computed via:
  - Strassen 2x2*2x2 (rank 7) for first two columns of B-block
  - Schoolbook 2x2*2x1 (rank 4) for third column of B-block
  Total per block: 11 rank terms.
4 blocks * 11 = 44 total rank.

The multiplication tensor T(A,B) = A*B where A is 4x4, B is 4x6, C is 4x6.
Decomposition: T = sum_r u_r * v_r * w_r
  u_r in R^{16} (A-space, index i*m+j)
  v_r in R^{24} (B-space, index j*p+k)
  w_r in R^{24} (C-space, index k*n+i per trace convention W[k*n+i, r])
"""

import time

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6

    U_rows = n * m   # 16
    V_rows = m * p   # 24
    W_rows = p * n   # 24

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def zeros(length):
        return [0] * length

    def strassen_2x2_2x2(ar, ac, br, bc, cr, cc):
        """Add 7 rank-1 terms for Strassen 2x2*2x2 at given offsets."""

        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)

        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(1, 0)] = 1; w[c_idx(1, 1)] = -1
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1
        v[b_idx(0, 1)] = 1; v[b_idx(1, 1)] = -1
        w[c_idx(0, 1)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 1)] = 1
        v[b_idx(1, 0)] = 1; v[b_idx(0, 0)] = -1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 0)] = 1
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(0, 1)] = 1
        v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = -1; w[c_idx(0, 1)] = 1
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(0, 0)] = -1
        v[b_idx(0, 0)] = 1; v[b_idx(0, 1)] = 1
        w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 1)] = 1; u[a_idx(1, 1)] = -1
        v[b_idx(1, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1
        add_term(u, v, w)

    def schoolbook_2x2_2x1(ar, ac, br, bc, cr, cc):
        """Add 4 rank-1 terms for schoolbook 2x2*2x1 at given offsets."""

        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)

        # A[0,0]*B[0,0] -> C[0,0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(0, 0)] = 1
        add_term(u, v, w)

        # A[0,1]*B[1,0] -> C[0,0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 1)] = 1
        v[b_idx(1, 0)] = 1
        w[c_idx(0, 0)] = 1
        add_term(u, v, w)

        # A[1,0]*B[0,0] -> C[1,0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(1, 0)] = 1
        add_term(u, v, w)

        # A[1,1]*B[1,0] -> C[1,0]
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 1)] = 1
        v[b_idx(1, 0)] = 1
        w[c_idx(1, 0)] = 1
        add_term(u, v, w)

    def product_2x2_2x3(ar, ac, br, bc, cr, cc):
        """Compute (2x2) A-block * (2x3) B-block = (2x3) C-block.
        Splits B into left 2x2 and right 2x1."""
        # Left part: A * B_left (2x2) via Strassen
        strassen_2x2_2x2(ar, ac, br, bc, cr, cc)
        # Right part: A * B_right (2x1) via schoolbook
        schoolbook_2x2_2x1(ar, ac, br, bc + 2, cr, cc + 2)

    # Block structure:
    # A = [[A11, A12], [A21, A22]]  each 2x2
    # B = [[B11, B12], [B21, B22]]  each 2x3
    # C = [[C11, C12], [C21, C22]]  each 2x3
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # C11 contributions
    product_2x2_2x3(0, 0, 0, 0, 0, 0)   # A11 * B11
    product_2x2_2x3(0, 2, 2, 0, 0, 0)   # A12 * B21

    # C12 contributions
    product_2x2_2x3(0, 0, 0, 3, 0, 3)   # A11 * B12
    product_2x2_2x3(0, 2, 2, 3, 0, 3)   # A12 * B22

    # C21 contributions
    product_2x2_2x3(2, 0, 0, 0, 2, 0)   # A21 * B11
    product_2x2_2x3(2, 2, 2, 0, 2, 0)   # A22 * B21

    # C22 contributions
    product_2x2_2x3(2, 0, 0, 3, 2, 3)   # A21 * B12
    product_2x2_2x3(2, 2, 2, 3, 2, 3)   # A22 * B22

    rank = len(U_cols)

    # Build output matrices: U is (U_rows x rank), V is (V_rows x rank), W is (W_rows x rank)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]

    return {"U": U, "V": V, "W": W}
