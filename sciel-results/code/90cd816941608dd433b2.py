"""Rewritten program: rank-33 decomposition for 2x3 * 3x6 matrix multiplication.

Approach:
  Split the inner dimension (3) into 2 + 1.
  A (2x3) = [A_2x2 | A_2x1]
  B (3x6) = [B_2x6 ; B_1x6]

  C = A_2x2 * B_2x6 + A_2x1 * B_1x6

  For A_2x2 * B_2x6:
    B_2x6 is split into three 2x2 blocks (cols 0-1, 2-3, 4-5).
    Each 2x2 * 2x2 product uses Strassen (7 multiplications).
    Total: 3 * 7 = 21 terms.

  For A_2x1 * B_1x6:
    This is a 2x1 times 1x6 product, giving a 2x6 result.
    C[i,k] = A[i,2] * B[2,k] for i in {0,1}, k in {0..5}.
    Each entry is a product of one A entry and one B entry.
    Total: 2 * 6 = 12 terms.

  Total rank: 21 + 12 = 33.

  The output convention is W[k*n + i, r] where n=2, p=6.
  So W has shape (p*n, rank) = (12, rank).
  U has shape (n*m, rank) = (6, rank).
  V has shape (m*p, rank) = (18, rank).
"""

DIMENSIONS = [2, 3, 6]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33

    col = 0

    # --- Strassen blocks for A_2x2 * B_2x6 ---
    # A_2x2 is A[0:2, 0:2], indices in U: i*3 + j for i in {0,1}, j in {0,1}
    #   U indices: 0 (a), 1 (b), 3 (d), 4 (e)
    # B_2x6 is B[0:2, :], indices in V: i*6 + k for i in {0,1}, k in {0..5}
    #   V indices: 0-5 (row 0), 6-11 (row 1)
    # C_2x6 is C[0:2, :], indices in W: k*2 + i for k in {0..5}, i in {0,1}
    #   W indices: 0-11

    # Strassen 2x2 * 2x2:
    # A = [[a, b], [d, e]], B = [[g, h], [m, n]]
    # Products:
    #   p1 = (a+e)(g+n)
    #   p2 = (d+e)g
    #   p3 = a(h-n)
    #   p4 = e(n-m)
    #   p5 = (a+d)(m+n)
    #   p6 = (b-e)(h+m)
    #   p7 = (a-b)(g+m)
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p5, p1-p2+p3+p6]]

    def add_strassen_block(b_col_start, c_col_start):
        nonlocal col
        # b_col_start: starting column in B_2x6 (0, 2, or 4)
        # c_col_start: starting column in C_2x6 (0, 2, or 4)
        # B entries: g=V[0*6+b_col_start], h=V[0*6+b_col_start+1]
        #             m=V[1*6+b_col_start], n=V[1*6+b_col_start+1]
        # C entries: C[0, c_col_start]=W[c_col_start*2+0], C[0, c_col_start+1]=W[c_col_start*2+1]
        #             C[1, c_col_start]=W[c_col_start*2+2], C[1, c_col_start+1]=W[c_col_start*2+3]

        # p1 = (a+e)(g+n)
        U[0][col] += 1   # a
        U[4][col] += 1   # e
        V[b_col_start][col] += 1       # g
        V[6 + b_col_start + 1][col] += 1  # n
        W[c_col_start][col] += 1       # C[0, c_col_start]
        W[c_col_start + 2][col] += 1   # C[1, c_col_start+1]
        col += 1

        # p2 = (d+e)g
        U[3][col] += 1   # d
        U[4][col] += 1   # e
        V[b_col_start][col] += 1  # g
        W[c_col_start + 2][col] += 1   # C[1, c_col_start]
        W[c_col_start + 3][col] -= 1   # C[1, c_col_start+1]
        col += 1

        # p3 = a(h-n)
        U[0][col] += 1   # a
        V[b_col_start + 1][col] += 1       # h
        V[6 + b_col_start + 1][col] -= 1   # -n
        W[c_col_start + 1][col] += 1       # C[0, c_col_start+1]
        W[c_col_start + 3][col] += 1       # C[1, c_col_start+1]
        col += 1

        # p4 = e(n-m)
        U[4][col] += 1   # e
        V[6 + b_col_start + 1][col] += 1  # n
        V[6 + b_col_start][col] -= 1      # -m
        W[c_col_start][col] += 1          # C[0, c_col_start]
        col += 1

        # p5 = (a+d)(m+n)
        U[0][col] += 1   # a
        U[3][col] += 1   # d
        V[6 + b_col_start][col] += 1      # m
        V[6 + b_col_start + 1][col] += 1  # n
        W[c_col_start][col] -= 1          # C[0, c_col_start]
        W[c_col_start + 1][col] += 1      # C[0, c_col_start+1]
        W[c_col_start + 2][col] += 1      # C[1, c_col_start]
        col += 1

        # p6 = (b-e)(h+m)
        U[1][col] += 1   # b
        U[4][col] -= 1   # -e
        V[b_col_start + 1][col] += 1      # h
        V[6 + b_col_start][col] += 1      # m
        W[c_col_start + 3][col] += 1      # C[1, c_col_start+1]
        col += 1

        # p7 = (a-b)(g+m)
        U[0][col] += 1   # a
        U[1][col] -= 1   # -b
        V[b_col_start][col] += 1      # g
        V[6 + b_col_start][col] += 1  # m
        W[c_col_start][col] += 1      # C[0, c_col_start]
        W[c_col_start + 2][col] += 1  # C[1, c_col_start]
        col += 1

    # Three Strassen blocks for B columns (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(2, 2)
    add_strassen_block(4, 4)

    # --- Schoolbook terms for A_2x1 * B_1x6 ---
    # A entries: A[0,2]=U[2], A[1,2]=U[5]
    # B entries: B[2,k]=V[12+k] for k in 0..5
    # C entries: C[i,k]=W[k*2+i] for i in {0,1}, k in 0..5
    for i in range(2):
        for k in range(6):
            U[i * 3 + 2][col] = 1   # A[i, 2]
            V[2 * 6 + k][col] = 1   # B[2, k]
            W[k * 2 + i][col] = 1   # C[i, k]
            col += 1

    return {"U": U, "V": V, "W": W}
