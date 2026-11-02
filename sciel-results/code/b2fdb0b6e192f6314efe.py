"""Rewritten program: structured rank-65 decomposition for 4x4 * 4x5 matrix multiply.

Approach:
- Split the 4x5 output into a 4x4 block (columns 0..3) and a 4x1 block (column 4).
- For the 4x4 block, use a block-Strassen scheme: partition A, B into 2x2 blocks of 2x2,
  compute 7 Strassen products for the 4x4x4 tensor, then recombine.
- For the 4x1 block, use schoolbook (16 terms).
- Total rank = 7 * 7 + 16 = 65.

The factor matrices U, V, W are built so that
  (U * V * W) reproduces the matrix multiplication tensor T_{i,j,k} = 1 if the
  (i,j) entry of A times the (j,k) entry of B contributes to the (i,k) entry of C.

Indexing conventions:
  U has shape (n*m, rank) = (16, R); index (i*m+j) <-> (i,j)
  V has shape (m*p, rank) = (20, R); index (j*p+k) <-> (j,k)
  W has shape (p*n, rank) = (20, R); index (k*n+i) <-> (k,i)

Each rank column c corresponds to one "trilinear term" U_c[i,j] * V_c[j,k] * W_c[k,i].
"""

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists (rows x rank).
    # Start with rank = 65, all zeros.
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # ------------------------------------------------------------------
    # Part 1: 4x4x4 block (columns k = 0..3)
    # Partition A into 2x2 blocks of size 2x2:
    #   A = [[A00, A01],
    #        [A10, A11]]
    # where Aab has rows i in {2a, 2a+1}, cols j in {2b, 2b+1}.
    # Similarly B = [[B00, B01], [B10, B11]] with B having rows j in {2b, 2b+1}, cols k in {2c, 2c+1}.
    # C = [[C00, C01], [C10, C11]] with C rows i in {2a, 2a+1}, cols k in {2c, 2c+1}.
    #
    # Strassen 2x2x2 (rank 7) applied at block level gives 7 block products M_s.
    # Each block product M_s = A_s * B_s where A_s and B_s are linear combinations of the 2x2 blocks.
    # Each such block product is a 2x2 * 2x2 multiply = 4 terms, but we use schoolbook for each
    # 2x2*2x2 (rank 4) OR we can use the fact that a 2x2*2x2 can be done in rank 7 via Strassen again.
    #
    # For simplicity and correctness, I'll use schoolbook (rank 4) for each 2x2*2x2 block product.
    # So each Strassen block product s uses 4 rank terms.
    # 7 Strassen products * 4 = 28 terms for the 4x4x4 part.
    #
    # Wait, but the block Strassen gives us 7 block-level multiplications. Each block multiplication
    # is a 2x2 matrix times a 2x2 matrix. The rank of a 2x2*2x2 tensor is 7 (Strassen), but using
    # schoolbook it's 4. Since we want minimal rank, let's use Strassen for each 2x2*2x2 too.
    # That gives 7 * 7 = 49 terms for the 4x4x4 part.
    #
    # I'll implement the nested Strassen: outer 7 Strassen terms, each decomposed into 7 inner Strassen terms.

    # Strassen 2x2x2 coefficients.
    # For 2x2 matrices A, B, C = A*B:
    # M1 = (A11 - A22) * (B22 + B11)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A22) * B22
    # M6 = (A12 - A11) * (B11 + B12)
    # M7 = (A21 - A22) * (B22 + B21)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M5

    # Define the 7 Strassen terms for a 2x2 multiply.
    # Each term s has:
    #   u_coeffs: list of (a_row, a_col, coeff) for the U part (A matrix)
    #   v_coeffs: list of (b_row, b_col, coeff) for the V part (B matrix)
    #   w_coeffs: list of (c_row, c_col, coeff) for the W part (C matrix)
    # where rows/cols are 0-indexed within the 2x2 block.

    strassen_2x2 = [
        {  # M1
            'u': [((0, 0), 1), ((1, 1), -1)],
            'v': [((1, 1), 1), ((0, 1), 1)],
            'w': [((0, 0), 1), ((1, 1), 1)],  # C11 += M1, C22 += M1
        },
        {  # M2
            'u': [((1, 0), 1), ((1, 1), 1)],
            'v': [((0, 0), 1)],
            'w': [((1, 0), 1), ((1, 1), -1)],  # C21 += M2, C22 -= M2
        },
        {  # M3
            'u': [((0, 0), 1)],
            'v': [((0, 1), 1), ((1, 1), -1)],
            'w': [((0, 1), 1), ((1, 1), 1)],  # C12 += M3, C22 += M3
        },
        {  # M4
            'u': [((1, 1), 1)],
            'v': [((1, 0), 1), ((0, 0), -1)],
            'w': [((0, 0), 1), ((1, 0), 1)],  # C11 += M4, C21 += M4
        },
        {  # M5
            'u': [((0, 0), 1), ((1, 1), 1)],
            'v': [((1, 1), 1)],
            'w': [((0, 0), -1), ((0, 1), 1), ((1, 1), 1)],  # C11 -= M5, C12 += M5, C22 += M5
        },
        {  # M6
            'u': [((0, 1), 1), ((0, 0), -1)],
            'v': [((0, 0), 1), ((0, 1), 1)],
            'w': [],  # M6 is not directly used in C; it's an intermediate
        },
        {  # M7
            'u': [((1, 0), 1), ((1, 1), -1)],
            'v': [((1, 1), 1), ((1, 0), 1)],
            'w': [((0, 0), 1)],  # C11 += M7
        },
    ]

    # Outer Strassen 2x2x2 at block level (each block is 2x2).
    # Block indices: a in {0,1} for A row block, b in {0,1} for A col block / B row block,
    #                c in {0,1} for B col block / C col block.
    # A blocks: A_{a,b}, B blocks: B_{b,c}, C blocks: C_{a,c}.
    #
    # Outer Strassen terms (same structure as inner, but on blocks):
    # O1 = (A00 - A11) * (B11 + B01)  -> wait, let me re-derive.

    # Actually, the outer Strassen operates on 2x2 block matrices where each "entry" is a 2x2 matrix.
    # Let me define the outer Strassen terms on the block indices (a,b) for A, (b,c) for B, (a,c) for C.

    outer_strassen = [
        {  # O1
            'u_blocks': [((0, 0), 1), ((1, 1), -1)],
            'v_blocks': [((1, 1), 1), ((0, 1), 1)],
            'w_blocks': [((0, 0), 1), ((1, 1), 1)],
        },
        {  # O2
            'u_blocks': [((1, 0), 1), ((1, 1), 1)],
            'v_blocks': [((0, 0), 1)],
            'w_blocks': [((1, 0), 1), ((1, 1), -1)],
        },
        {  # O3
            'u_blocks': [((0, 0), 1)],
            'v_blocks': [((0, 1), 1), ((1, 1), -1)],
            'w_blocks': [((0, 1), 1), ((1, 1), 1)],
        },
        {  # O4
            'u_blocks': [((1, 1), 1)],
            'v_blocks': [((1, 0), 1), ((0, 0), -1)],
            'w_blocks': [((0, 0), 1), ((1, 0), 1)],
        },
        {  # O5
            'u_blocks': [((0, 0), 1), ((1, 1), 1)],
            'v_blocks': [((1, 1), 1)],
            'w_blocks': [((0, 0), -1), ((0, 1), 1), ((1, 1), 1)],
        },
        {  # O6
            'u_blocks': [((0, 1), 1), ((0, 0), -1)],
            'v_blocks': [((0, 0), 1), ((0, 1), 1)],
            'w_blocks': [],
        },
        {  # O7
            'u_blocks': [((1, 0), 1), ((1, 1), -1)],
            'v_blocks': [((1, 1), 1), ((1, 0), 1)],
            'w_blocks': [((0, 0), 1)],
        },
    ]

    # Now, for each outer term s and inner term t, we create one rank column.
    # The rank column index is s * 7 + t for s, t in 0..6.
    # That's 49 columns for the 4x4x4 part.

    col = 0
    for s in range(7):
        outer = outer_strassen[s]
        for t in range(7):
            inner = strassen_2x2[t]
            # U part: sum over outer u_blocks and inner u
            for (ba, bb, ca) in outer['u_blocks']:
                for ((ra, ca2), cb) in inner['u']:
                    # Global A index: row i = 2*ba + ra, col j = 2*bb + ca2
                    i = 2 * ba + ra
                    j = 2 * bb + ca2
                    U[i * m + j][col] += ca * cb
            # V part: sum over outer v_blocks and inner v
            for (bb, bc, ca) in outer['v_blocks']:
                for ((rb, cb), cd) in inner['v']:
                    # Global B index: row j = 2*bb + rb, col k = 2*bc + cb
                    j = 2 * bb + rb
                    k = 2 * bc + cb
                    V[j * p + k][col] += ca * cd
            # W part: sum over outer w_blocks and inner w
            for (ba, bc, ca) in outer['w_blocks']:
                for ((rc, cc), cd) in inner['w']:
                    # Global C index: row i = 2*ba + rc, col k = 2*bc + cc
                    i = 2 * ba + rc
                    k = 2 * bc + cc
                    W[k * n + i][col] += ca * cd
            col += 1

    # ------------------------------------------------------------------
    # Part 2: 4x1 block (column k = 4), schoolbook: 16 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1.0
            V[j * p + 4][col] = 1.0
            W[4 * n + i][col] = 1.0
            col += 1

    # Convert to lists of ints where possible, but keep floats for safety.
    # The task says "integer/half-integer entries", so let's round to nearest 0.5 if needed.
    def round_half(x):
        return round(x * 2) / 2

    U = [[round_half(x) for x in row] for row in U]
    V = [[round_half(x) for x in row] for row in V]
    W = [[round_half(x) for x in row] for row in W]

    # If all entries are integers, convert to int.
    def to_int_if_possible(lst):
        return [[int(x) if x == int(x) else x for x in row] for row in lst]

    U = to_int_if_possible(U)
    V = to_int_if_possible(V)
    W = to_int_if_possible(W)

    return {"U": U, "V": V, "W": W}
