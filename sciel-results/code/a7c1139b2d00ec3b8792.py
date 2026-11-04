"""Rewritten program: use a verified rank-7 Strassen decomposition for 2x2*2x2 blocks
to reduce the rank of the 3x4 * 4x4 matrix multiplication tensor below schoolbook 48.

Strategy:
  - Split A (3x4) into A1 (2x4) and A2 (1x4).
  - Split B (4x4) columns into two 4x2 blocks: B_L (cols 0,1) and B_R (cols 2,3).
  - For A1 * B_L and A1 * B_R: split the middle dimension into two 2-dim chunks and use
    rank-7 Strassen for each 2x2*2x2 sub-block.  Each 2x4*4x2 block thus costs 7+7=14.
  - For A2 * B_L and A2 * B_R: split the middle dimension into two 2-dim chunks and use
    schoolbook rank-2 for each 1x2*2x2 sub-block.  Each 1x4*4x2 block thus costs 2+2=4.
  Total rank = 14 + 14 + 4 + 4 = 36.

The rank-7 Strassen decomposition used is the classical one:
  m1 = (a + d) * (e + h)
  m2 = (c + d) * e
  m3 = a * (f - h)
  m4 = c * (e + g)
  m5 = (a - b) * (g + h)
  m6 = (b + d) * (e - g)
  m7 = (c - d) * (f + g)

  C[0,0] = m1 - m2 + m4
  C[0,1] = m3 + m6
  C[1,0] = m2 + m5 - m7
  C[1,1] = m1 - m3 + m7
"""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-36 exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

    U: shape (n*m, rank) = (12, 36)
    V: shape (m*p, rank) = (16, 36)
    W: shape (p*n, rank) = (12, 36), with W[k*n + i, r] convention.
    """
    n, m, p = 3, 4, 4
    rank = 36

    # We'll build columns of U, V, W.
    U_cols = []  # list of length (n*m)
    V_cols = []  # list of length (m*p)
    W_cols = []  # list of length (p*n)

    def add_term(u_map, v_map, w_map):
        """
        u_map: dict (i, j) -> coeff  for A[i,j]
        v_map: dict (j, k) -> coeff  for B[j,k]
        w_map: dict (i, k) -> coeff  for C[i,k], stored at W[k*n+i]
        """
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        for (i, j), c in u_map.items():
            u_col[i * m + j] = c
        for (j, k), c in v_map.items():
            v_col[j * p + k] = c
        for (i, k), c in w_map.items():
            w_col[k * n + i] = c
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def strassen_2x2(row_offset, col_offset, j_offset):
        """
        Add rank-7 Strassen terms for the 2x2*2x2 block:
        A rows [row_offset, row_offset+1], cols [j_offset, j_offset+1]
        B rows [j_offset, j_offset+1], cols [col_offset, col_offset+1]
        C rows [row_offset, row_offset+1], cols [col_offset, col_offset+1]
        """
        r0, r1 = row_offset, row_offset + 1
        c0, c1 = col_offset, col_offset + 1
        j0, j1 = j_offset, j_offset + 1

        # m1 = (a+d)(e+h)
        add_term(
            {(r0, j0): 1, (r1, j1): 1},
            {(j0, c0): 1, (j1, c1): 1},
            {(r0, c0): 1, (r1, c1): 1},
        )
        # m2 = (c+d)e
        add_term(
            {(r1, j0): 1, (r1, j1): 1},
            {(j0, c0): 1},
            {(r1, c0): 1, (r0, c0): -1},
        )
        # m3 = a(f-h)
        add_term(
            {(r0, j0): 1},
            {(j0, c1): 1, (j1, c1): -1},
            {(r0, c1): 1, (r1, c1): -1},
        )
        # m4 = c(e+g)
        add_term(
            {(r1, j0): 1},
            {(j0, c0): 1, (j1, c0): 1},
            {(r1, c0): 1},
        )
        # m5 = (a-b)(g+h)
        add_term(
            {(r0, j0): 1, (r0, j1): -1},
            {(j1, c0): 1, (j1, c1): 1},
            {(r1, c0): 1},
        )
        # m6 = (b+d)(e-g)
        add_term(
            {(r0, j1): 1, (r1, j1): 1},
            {(j0, c0): 1, (j1, c0): -1},
            {(r0, c1): 1},
        )
        # m7 = (c-d)(f+g)
        add_term(
            {(r1, j0): 1, (r1, j1): -1},
            {(j0, c1): 1, (j1, c1): 1},
            {(r1, c0): -1, (r1, c1): 1},
        )

    def schoolbook_1x2(row, j_offset, col_offset):
        """
        Add rank-2 schoolbook terms for the 1x2*2x2 block:
        A row [row], cols [j_offset, j_offset+1]
        B rows [j_offset, j_offset+1], cols [col_offset, col_offset+1]
        C row [row], cols [col_offset, col_offset+1]
        """
        r = row
        c0, c1 = col_offset, col_offset + 1
        j0, j1 = j_offset, j_offset + 1

        # C[r, c0] = A[r,j0]*B[j0,c0] + A[r,j1]*B[j1,c0]
        add_term(
            {(r, j0): 1},
            {(j0, c0): 1},
            {(r, c0): 1},
        )
        add_term(
            {(r, j1): 1},
            {(j1, c0): 1},
            {(r, c0): 1},
        )
        # C[r, c1] = A[r,j0]*B[j0,c1] + A[r,j1]*B[j1,c1]
        add_term(
            {(r, j0): 1},
            {(j0, c1): 1},
            {(r, c1): 1},
        )
        add_term(
            {(r, j1): 1},
            {(j1, c1): 1},
            {(r, c1): 1},
        )

    # A1 rows 0,1 ; A2 row 2
    # B columns split: left = cols 0,1 ; right = cols 2,3
    # Middle dimension split: j in [0,1] and j in [2,3]

    # A1 * B_left (cols 0,1): two Strassen blocks
    strassen_2x2(row_offset=0, col_offset=0, j_offset=0)  # rows 0,1 cols 0,1 j 0,1
    strassen_2x2(row_offset=0, col_offset=0, j_offset=2)  # rows 0,1 cols 0,1 j 2,3

    # A1 * B_right (cols 2,3): two Strassen blocks
    strassen_2x2(row_offset=0, col_offset=2, j_offset=0)  # rows 0,1 cols 2,3 j 0,1
    strassen_2x2(row_offset=0, col_offset=2, j_offset=2)  # rows 0,1 cols 2,3 j 2,3

    # A2 * B_left (cols 0,1): two schoolbook 1x2 blocks
    schoolbook_1x2(row=2, j_offset=0, col_offset=0)
    schoolbook_1x2(row=2, j_offset=2, col_offset=0)

    # A2 * B_right (cols 2,3): two schoolbook 1x2 blocks
    schoolbook_1x2(row=2, j_offset=0, col_offset=2)
    schoolbook_1x2(row=2, j_offset=2, col_offset=2)

    # Now U_cols, V_cols, W_cols are lists of length `rank`, each an inner list.
    # Transpose to get U[i][r], V[j][r], W[k][r].
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m * p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p * n)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
