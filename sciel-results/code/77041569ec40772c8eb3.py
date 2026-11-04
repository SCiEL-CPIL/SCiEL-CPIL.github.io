"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Split B into first 4 columns and last 2 columns.
- First 4 cols: 4x4*4x4 via 4 Strassen 2x2*2x2 (rank 28)
- Last 2 cols: 4x4*4x2 via schoolbook (rank 32)
Total rank: 60

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

    def strassen_2x2_2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """Add 7 rank-1 terms for Strassen 2x2*2x2.
        
        a_row_off, a_col_off: offset in A (4x4)
        b_row_off, b_col_off: offset in B (4x6) 
        c_row_off, c_col_off: offset in C (4x6)
        """
        def ai(i, j): return (a_row_off + i) * m + (a_col_off + j)
        def bi(i, j): return (b_row_off + i) * p + (b_col_off + j)
        def ci(i, j): return (c_col_off + j) * n + (c_row_off + i)

        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(0, 0)] = 1; u[ai(1, 1)] = 1
        v[bi(0, 0)] = 1; v[bi(1, 1)] = 1
        w[ci(0, 0)] = 1; w[ci(1, 1)] = 1
        add_term(u, v, w)

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(1, 0)] = 1; u[ai(1, 1)] = 1
        v[bi(0, 0)] = 1
        w[ci(1, 0)] = 1; w[ci(1, 1)] = -1
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(0, 0)] = 1
        v[bi(0, 1)] = 1; v[bi(1, 1)] = -1
        w[ci(0, 1)] = 1; w[ci(1, 1)] = 1
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(1, 1)] = 1
        v[bi(1, 0)] = 1; v[bi(0, 0)] = -1
        w[ci(0, 0)] = 1; w[ci(1, 0)] = 1
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(0, 0)] = 1; u[ai(0, 1)] = 1
        v[bi(1, 1)] = 1
        w[ci(0, 0)] = -1; w[ci(0, 1)] = 1
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(1, 0)] = 1; u[ai(0, 0)] = -1
        v[bi(0, 0)] = 1; v[bi(0, 1)] = 1
        w[ci(1, 1)] = 1
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[ai(0, 1)] = 1; u[ai(1, 1)] = -1
        v[bi(1, 0)] = 1; v[bi(1, 1)] = 1
        w[ci(0, 0)] = 1
        add_term(u, v, w)

    def schoolbook_4x4_4x2():
        """Add 32 rank-1 terms for schoolbook 4x4*4x2 (last 2 columns)."""
        for i in range(4):       # row of C
            for j in range(4):   # col of A / row of B
                for k in range(2):  # col of B (4,5) / col of C (4,5)
                    u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
                    u[i * m + j] = 1
                    v[j * p + (k + 4)] = 1
                    w[(k + 4) * n + i] = 1
                    add_term(u, v, w)

    # First 4 columns of B (cols 0-3): 4x4 * 4x4 via Strassen
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12], [B21, B22]] each 2x2 (cols 0-3)
    # C = [[C11, C12], [C21, C22]] each 2x2 (cols 0-3)
    
    # C11 = A11*B11 + A12*B21
    strassen_2x2_2x2(0, 0, 0, 0, 0, 0)   # A11 * B11
    strassen_2x2_2x2(0, 2, 2, 0, 0, 0)   # A12 * B21

    # C12 = A11*B12 + A12*B22
    strassen_2x2_2x2(0, 0, 0, 2, 0, 2)   # A11 * B12
    strassen_2x2_2x2(0, 2, 2, 2, 0, 2)   # A12 * B22

    # C21 = A21*B11 + A22*B21
    strassen_2x2_2x2(2, 0, 0, 0, 2, 0)   # A21 * B11
    strassen_2x2_2x2(2, 2, 2, 0, 2, 0)   # A22 * B21

    # C22 = A21*B12 + A22*B22
    strassen_2x2_2x2(2, 0, 0, 2, 2, 2)   # A21 * B12
    strassen_2x2_2x2(2, 2, 2, 2, 2, 2)   # A22 * B22

    # Last 2 columns of B (cols 4-5): 4x4 * 4x2 via schoolbook
    schoolbook_4x4_4x2()

    rank = len(U_cols)

    # Build output matrices: U is (U_rows x rank), V is (V_rows x rank), W is (W_rows x rank)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]

    return {"U": U, "V": V, "W": W}
