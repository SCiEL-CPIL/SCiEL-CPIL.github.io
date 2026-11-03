"""Winograd Fourier-based approach for 2x4x5 matrix multiplication tensor."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5  # A is n x m, B is m x p, C is n x p
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]

    # We'll use a rank-36 decomposition based on:
    # Split A into 2 blocks of 2x2 (along columns)
    # Split B into 2 blocks of 2x5 (along rows)
    # For each pair (A_block, B_block), compute 2x2 * 2x5
    # 2x2 * 2x5 can be done with rank 18:
    #   Split 2x5 B into 2x2 + 2x2 + 2x1
    #   2x2*2x2 via Strassen (rank 7)
    #   2x2*2x2 via Strassen (rank 7)
    #   2x2*2x1 via schoolbook (rank 4)
    # Total: 2 * 18 = 36

    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 36

    col = [0]  # use list for mutable closure

    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term defined by sparse dictionaries."""
        if col[0] >= rank:
            raise ValueError(f"Rank exceeded: col={col[0]}, rank={rank}")
        c = col[0]
        for idx, val in u_dict.items():
            U[idx][c] += val
        for idx, val in v_dict.items():
            V[idx][c] += val
        for idx, val in w_dict.items():
            W[idx][c] += val
        col[0] += 1

    def a_idx(i, j):
        """Global index for A[i,j] in U."""
        return i * m + j

    def b_idx(j, k):
        """Global index for B[j,k] in V."""
        return j * p + k

    def c_idx(k, i):
        """Global index for C[i,k] in W (trace convention W[k*n+i])."""
        return k * n + i

    # Strassen 2x2 * 2x2: adds 7 rank-1 terms
    # a00,a01,a10,a11 are global A indices
    # b00,b01,b10,b11 are global B indices
    # c00,c01,c10,c11 are global C indices (for output C[i,j] where i=0,1; j=0,1)
    def strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})

    # Schoolbook 2x2 * 2x1: adds 4 rank-1 terms
    # a00,a01,a10,a11 are global A indices
    # b0,b1 are global B indices (B is 2x1)
    # c0,c1 are global C indices (C is 2x1)
    def schoolbook_2x2_1(a00, a01, a10, a11, b0, b1, c0, c1):
        # C0 = A00*B0 + A01*B1
        add_term({a00: 1}, {b0: 1}, {c0: 1})
        add_term({a01: 1}, {b1: 1}, {c0: 1})
        # C1 = A10*B0 + A11*B1
        add_term({a10: 1}, {b0: 1}, {c1: 1})
        add_term({a11: 1}, {b1: 1}, {c1: 1})

    # Now build the full decomposition
    # A is 2x4, B is 4x5, C is 2x5
    # Split A into A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into B_T (rows 0,1) and B_B (rows 2,3)
    # C = A_L * B_T + A_R * B_B

    # Block 1: A_L (2x2) * B_T (2x5)
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1]
    # B_T: rows 0,1; cols 0,1,2,3,4
    # Split B_T into B_T1 (cols 0,1), B_T2 (cols 2,3), B_T3 (col 4)

    # A_L * B_T1 (2x2 * 2x2)
    strassen_2x2(
        a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1),
        b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1),
        c_idx(0,0), c_idx(1,0), c_idx(0,1), c_idx(1,1)
    )

    # A_L * B_T2 (2x2 * 2x2)
    strassen_2x2(
        a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1),
        b_idx(0,2), b_idx(0,3), b_idx(1,2), b_idx(1,3),
        c_idx(2,0), c_idx(3,0), c_idx(2,1), c_idx(3,1)
    )

    # A_L * B_T3 (2x2 * 2x1)
    schoolbook_2x2_1(
        a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1),
        b_idx(0,4), b_idx(1,4),
        c_idx(4,0), c_idx(4,1)
    )

    # Block 2: A_R (2x2) * B_B (2x5)
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3]
    # B_B: rows 2,3; cols 0,1,2,3,4
    # Split B_B into B_B1 (cols 0,1), B_B2 (cols 2,3), B_B3 (col 4)

    # A_R * B_B1 (2x2 * 2x2)
    strassen_2x2(
        a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3),
        b_idx(2,0), b_idx(2,1), b_idx(3,0), b_idx(3,1),
        c_idx(0,0), c_idx(1,0), c_idx(0,1), c_idx(1,1)
    )

    # A_R * B_B2 (2x2 * 2x2)
    strassen_2x2(
        a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3),
        b_idx(2,2), b_idx(2,3), b_idx(3,2), b_idx(3,3),
        c_idx(2,0), c_idx(3,0), c_idx(2,1), c_idx(3,1)
    )

    # A_R * B_B3 (2x2 * 2x1)
    schoolbook_2x2_1(
        a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3),
        b_idx(2,4), b_idx(3,4),
        c_idx(4,0), c_idx(4,1)
    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
