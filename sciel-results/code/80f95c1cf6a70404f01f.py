"""
Matrix multiplication 2x2 @ 2x8 using Strassen's 7-multiply algorithm
for each of the four 2x2 block-products.

Decomposition:
  A (2x2) = [[A00, A01],
             [A10, A11]]
  B (2x8) = [[B00, B01, B02, B03, B04, B05, B06, B07],
             [B10, B11, B12, B13, B14, B15, B16, B17]]

  Split B into four 2x2 blocks:
  B = [B_a, B_b, B_c, B_d] where each block is 2x2:
  B_a = [[B00, B01], [B10, B11]]
  B_b = [[B02, B03], [B12, B13]]
  B_c = [[B04, B05], [B14, B15]]
  B_d = [[B06, B07], [B16, B17]]

  C = A @ B = [C_a, C_b, C_c, C_d] where each C_x is 2x2:
  C_a = A @ B_a, etc.

  For each block x in {a,b,c,d}, use Strassen:
  M1 = (A00 + A11) @ (Bx00 + Bx11)
  M2 = (A10 + A11) @ Bx00
  M3 = A00 @ (Bx10 - Bx11)
  M4 = A11 @ (Bx00 - Bx10)
  M5 = (A00 + A01) @ Bx11
  M6 = (A10 - A00) @ (Bx00 + Bx10)
  M7 = (A01 - A11) @ (Bx11 + Bx01)

  Cx00 = M1 + M4 - M5 + M7
  Cx01 = M3 + M5
  Cx10 = M2 + M4
  Cx11 = M1 - M2 + M3 + M6

  Total rank = 4 * 7 = 28.
"""

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    # U: shape (n*m, rank) = (4, 28)
    # V: shape (m*p, rank) = (16, 28)
    # W: shape (p*n, rank) = (16, 28)

    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    # Strassen multipliers s = 1..7 (0-indexed 0..6)
    # For each block x = 0..3 (columns 2x, 2x+1 of B)
    # and each multiplier s, we have one rank-1 term.
    # Column index r = 4*s + x (0-indexed), so r in 0..27.

    # Strassen U vectors (4 components: A00, A01, A10, A11)
    # A is 2x2, flattened row-major: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    U_strassen = [
        [1,  0,  0,  1],  # s=0: A00 + A11
        [0,  0,  1,  1],  # s=1: A10 + A11
        [1,  0,  0,  0],  # s=2: A00
        [0,  0,  0,  1],  # s=3: A11
        [1,  1,  0,  0],  # s=4: A00 + A01
        [-1, 0,  1,  0],  # s=5: A10 - A00
        [-1, 1,  0,  1],  # s=6: A01 - A11
    ]

    # Strassen V vectors for a 2x2 block Bx.
    # Bx flattened row-major: Bx[0]=Bx00, Bx[1]=Bx01, Bx[2]=Bx10, Bx[3]=Bx11
    V_strassen = [
        [1,  0,  0,  1],  # s=0: Bx00 + Bx11
        [1,  0,  0,  0],  # s=1: Bx00
        [0,  0,  1, -1],  # s=2: Bx10 - Bx11
        [1,  0, -1, 0],   # s=3: Bx00 - Bx10
        [0,  0,  0,  1],  # s=4: Bx11
        [1,  1,  0,  0],  # s=5: Bx00 + Bx10
        [0,  1,  0,  1],  # s=6: Bx11 + Bx01
    ]

    # Strassen W vectors for a 2x2 block Cx.
    # Cx flattened in the convention W[k*n+i, r] for local 2x2:
    # Local indices: k_loc in {0,1} (column of Cx), i_loc in {0,1} (row of Cx)
    # Local flat index = k_loc * 2 + i_loc
    # Cx00: i_loc=0, k_loc=0 -> idx 0
    # Cx01: i_loc=0, k_loc=1 -> idx 1
    # Cx10: i_loc=1, k_loc=0 -> idx 2
    # Cx11: i_loc=1, k_loc=1 -> idx 3
    W_strassen = [
        [1,  0,  0,  1],  # s=0: M1 contributes to Cx00 and Cx11
        [0,  0,  1, -1],  # s=1: M2 contributes to Cx10 and Cx11 (negative)
        [0,  1,  0,  1],  # s=2: M3 contributes to Cx01 and Cx11
        [1,  0,  1,  0],  # s=3: M4 contributes to Cx00 and Cx10
        [-1, 1,  0,  0],  # s=4: M5 contributes to Cx00 (neg) and Cx01
        [0,  0,  0,  1],  # s=5: M6 contributes to Cx11
        [1,  0,  0,  0],  # s=6: M7 contributes to Cx00
    ]

    for x in range(4):  # block index 0..3
        for s in range(7):  # Strassen multiplier 0..6
            r = 4 * x + s  # column index in 0..27

            # U vector
            for a_idx in range(4):
                U[a_idx][r] = U_strassen[s][a_idx]

            # V vector: block x corresponds to columns 2x, 2x+1 of B
            # B is 2x8, flattened row-major: B[j*8 + k]
            # For block x, local row j_loc in {0,1}, local col k_loc in {0,1}
            # Global: j = j_loc, k = 2*x + k_loc
            # B index = j * 8 + (2*x + k_loc)
            for j_loc in range(2):
                for k_loc in range(2):
                    b_idx = j_loc * 8 + (2 * x + k_loc)
                    v_idx = j_loc * 2 + k_loc  # local Bx index
                    V[b_idx][r] = V_strassen[s][v_idx]

            # W vector: output C is 2x8, flattened W[k*n + i, r] where k is col of C, i is row
            # For block x, C block is 2x2 at columns 2x, 2x+1
            # Local: i_loc in {0,1} (row), k_loc in {0,1} (col within block)
            # Global: i = i_loc, k = 2*x + k_loc
            # W index = k * n + i = (2*x + k_loc) * 2 + i_loc
            for i_loc in range(2):
                for k_loc in range(2):
                    w_idx = (2 * x + k_loc) * 2 + i_loc
                    c_idx = k_loc * 2 + i_loc  # local Cx index
                    W[w_idx][r] = W_strassen[s][c_idx]

    return {"U": U, "V": V, "W": W}
